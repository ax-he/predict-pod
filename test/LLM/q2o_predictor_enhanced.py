# q2o_predictor.py (Enhanced Version)
# 目标：输入自然语言“问题” -> 自动估算输入 token S、问题类型、输出 token O 的 p50/p90
# 主要改进：
# 1. 增强长度提示解析，支持多种语言和单位
# 2. 优化任务类型检测逻辑，避免规则冲突
# 3. 加强错误处理和边界防护
# 4. 添加批量处理、缓存、置信度评分等实用功能

import os
import re
import math
import json
import argparse
import logging
from typing import Tuple, Dict, Optional, List, Union
from functools import lru_cache
from dataclasses import dataclass

import numpy as np
import pandas as pd

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 可选依赖：tiktoken、transformers、scikit-learn
try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False
    logger.warning("tiktoken not available, falling back to other tokenizers")

try:
    from transformers import AutoTokenizer
    _HAS_HF = True
except ImportError:
    _HAS_HF = False
    logger.warning("transformers not available, cannot use HF tokenizers")

try:
    from sklearn.linear_model import LinearRegression, QuantileRegressor
    from sklearn.preprocessing import OneHotEncoder
    _HAS_SK = True
except ImportError:
    _HAS_SK = False
    logger.warning("scikit-learn not available, using prior-based prediction only")


# --------------------------
# 数据类定义
# --------------------------
@dataclass
class PredictionResult:
    """预测结果数据类"""
    S: int
    task: str
    tokenizer: str
    O_mean: float
    O_p50: float
    O_p90: float
    confidence: float
    used_len_hint: bool = False
    len_hint_value: Optional[int] = None


# --------------------------
# 1) 增强版长度提示解析
# --------------------------
class LengthHintParser:
    """解析各种形式的长度提示"""
    
    @staticmethod
    def parse_len_hint(question: str) -> Optional[Tuple[int, str]]:
        """
        从问题中解析长度提示，返回 (token估算值, 匹配的单位)
        支持多种语言和单位
        """
        if not question:
            return None
        
        # 定义匹配模式和对应的token转换系数
        patterns = [
            # 中文单位
            (r"(\d+)\s*字", "字", 1.0),           # 100字 (1字≈1token)
            (r"(\d+)\s*词", "词", 1.0),           # 100词 (1词≈1token)
            (r"(\d+)\s*句", "句", 15.0),          # 3句话 (假设1句≈15token)
            (r"(\d+)\s*段", "段", 100.0),         # 2段落 (假设1段≈100token)
            (r"(\d+)\s*行", "行", 20.0),          # 5行 (假设1行≈20token)
            (r"(\d+)\s*页", "页", 500.0),         # 1页 (假设1页≈500token)
            
            # 英文单位
            (r"(\d+)\s*words?", "words", 1.0),           # 100 words
            (r"(\d+)\s*sentences?", "sentences", 15.0),  # 3 sentences
            (r"(\d+)\s*paragraphs?", "paragraphs", 100.0), # 2 paragraphs
            (r"(\d+)\s*lines?", "lines", 20.0),          # 5 lines
            (r"(\d+)\s*pages?", "pages", 500.0),         # 1 page
            
            # 范围表示
            (r"(\d+)\s*到\s*(\d+)\s*字", "字范围", 1.0),    # 100到200字
            (r"(\d+)\s*-\s*(\d+)\s*words?", "word范围", 1.0), # 100-200 words
            
            # 约/大概等模糊表述
            (r"(?:约|大概|大约|左右)\s*(\d+)\s*字", "约字", 1.0),
            (r"(?:about|approximately|around)\s*(\d+)\s*words?", "约words", 1.0),
        ]
        
        question_lower = question.lower()
        
        for pattern, unit, coefficient in patterns:
            matches = re.finditer(pattern, question_lower, re.IGNORECASE)
            for match in matches:
                try:
                    if "范围" in unit or "-" in pattern:
                        # 处理范围：取平均值
                        start = int(match.group(1))
                        end = int(match.group(2))
                        avg_value = (start + end) / 2
                        token_estimate = int(avg_value * coefficient)
                        return token_estimate, f"{unit}({start}-{end})"
                    else:
                        # 处理单个数值
                        num = int(match.group(1))
                        token_estimate = int(num * coefficient)
                        return token_estimate, unit
                except (ValueError, IndexError):
                    continue
        
        return None
    
    @staticmethod
    def has_length_constraint(question: str) -> bool:
        """快速检查是否包含长度约束提示"""
        constraint_keywords = [
            '字', '词', '句', '段', '行', '页',
            'word', 'sentence', 'paragraph', 'line', 'page',
            '约', '大概', '左右', 'about', 'approximately', 'around'
        ]
        return any(keyword in question.lower() for keyword in constraint_keywords)


# --------------------------
# 2) 估算输入 token 数 S
# --------------------------
class TokenCounter:
    def __init__(self,
                 prefer_hf_model: Optional[str] = None,
                 use_chat_template: bool = False):
        """
        prefer_hf_model: 显式指定 HF tokenizer 名（优先级最高）
        use_chat_template: 若 True，按 chat 模板包装后计数（更贴近你线上 Chat 模式）
        """
        self.enc = None
        self.hf_tok = None
        self.model_name = None
        self.use_chat_template = use_chat_template
        self.errors = []

        # 1) 优先：尝试加载 DeepSeek R1 7B 或用户指定的 HF tokenizer
        if _HAS_HF:
            candidates = self._get_tokenizer_candidates(prefer_hf_model)
            for name in candidates:
                try:
                    logger.info(f"尝试加载tokenizer: {name}")
                    self.hf_tok = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=True)
                    self.model_name = name
                    logger.info(f"成功加载tokenizer: {name}")
                    break
                except Exception as e:
                    logger.warning(f"加载tokenizer {name} 失败: {e}")
                    continue

        # 2) 次选：tiktoken(cl100k_base)
        if self.hf_tok is None and _HAS_TIKTOKEN:
            try:
                self.enc = tiktoken.get_encoding("cl100k_base")
                logger.info("使用tiktoken(cl100k_base)")
            except Exception as e:
                self.enc = None
                self.errors.append(f"Tiktoken failed: {e}")

        # 3) 末选：轻量级HF tokenizer
        if self.hf_tok is None and self.enc is None and _HAS_HF:
            try:
                self.hf_tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
                self.model_name = "gpt2"
                logger.info("回退到gpt2 tokenizer")
            except Exception as e:
                self.hf_tok = None
                self.errors.append(f"GPT2 tokenizer failed: {e}")

        if self.hf_tok is None and self.enc is None:
            logger.warning("所有tokenizer都失败，将使用字符近似法")

    def _get_tokenizer_candidates(self, prefer_hf_model: Optional[str]) -> List[str]:
        """获取tokenizer候选列表"""
        candidates = []
        if prefer_hf_model:
            candidates.append(prefer_hf_model)
        
        # 环境变量兜底
        env_vars = ["Q2O_TOKENIZER", "HF_TOKENIZER", "TOKENIZER_NAME"]
        for env_var in env_vars:
            env_val = os.getenv(env_var)
            if env_val:
                candidates.append(env_val)
        
        # 常见 DeepSeek 名称兜底
        candidates.extend([
            "deepseek-ai/DeepSeek-R1-7B",
            "deepseek-ai/DeepSeek-R1",
            "deepseek-ai/DeepSeek-V2-7B",
            "deepseek-ai/DeepSeek-V2",
            "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            "deepseek-ai/DeepSeek-LLM-7B-Chat",
        ])
        
        return candidates

    def count(self, text: str) -> int:
        """计算token数量，带有健壮的错误处理"""
        if not text or not isinstance(text, str):
            return 1
        
        text = text.strip()
        if not text:
            return 1
        
        errors = []
        
        # HF tokenizer 分支
        if self.hf_tok is not None:
            try:
                if self.use_chat_template:
                    messages = [{"role": "user", "content": text}]
                    prompt = self.hf_tok.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    tokens = self.hf_tok(prompt, add_special_tokens=False).input_ids
                else:
                    tokens = self.hf_tok.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                errors.append(f"HF tokenizer failed: {e}")
                # 尝试不使用chat template
                try:
                    tokens = self.hf_tok.encode(text, add_special_tokens=False)
                    return len(tokens)
                except Exception as e2:
                    errors.append(f"HF fallback also failed: {e2}")

        # tiktoken 分支
        if self.enc is not None:
            try:
                return len(self.enc.encode(text))
            except Exception as e:
                errors.append(f"Tiktoken failed: {e}")

        # 记录错误日志
        if errors:
            logger.warning(f"Tokenizer errors: {'; '.join(errors)}")
        
        # 极简退化：按字符估算（混合文本 1 token≈3.5 chars）
        char_count = len(text)
        token_estimate = max(1, int(math.ceil(char_count / 3.5)))
        logger.info(f"使用字符近似法: {char_count} chars -> {token_estimate} tokens")
        
        return token_estimate


# --------------------------
# 3) 优化版任务类型识别
# --------------------------
class TaskTypeDetector:
    """优化版任务类型检测器"""
    
    def __init__(self):
        # 按优先级从高到低排列，避免使用兜底规则
        self.type_rules = [
            ("code", self._is_code_question),
            ("math", self._is_math_question),
            ("translate", self._is_translate_question),
            ("summarize", self._is_summarize_question),
            ("qa_explain", self._is_explanation_question),
            ("creative", self._is_creative_question),
            ("qa_short", self._is_short_question),
        ]
    
    def detect(self, question: str) -> str:
        """检测问题类型"""
        if not question:
            return "qa_short"
            
        q = question.lower().strip()
        
        for task_type, check_func in self.type_rules:
            if check_func(q, question):
                return task_type
        
        # 最终兜底逻辑
        return self._get_fallback_type(q)
    
    def _is_code_question(self, q_lower: str, original: str) -> bool:
        code_keywords = [
            r"\b(code|程序|编程|实现|函数|算法|bug|错误|调试|python|java|cpp|c\+\+|javascript|js|html|css)\b",
            r"\b(编程|写代码|代码实现|程序实现|debug|报错|异常|stack trace)\b",
            r"\b(def |function |class |import |print |return )\b"
        ]
        return any(re.search(pattern, q_lower, re.IGNORECASE) for pattern in code_keywords)
    
    def _is_math_question(self, q_lower: str, original: str) -> bool:
        math_keywords = [
            r"\b(计算|数学|公式|方程|函数|导数|积分|概率|统计|几何|代数)\b",
            r"\b(calculate|compute|solve|equation|formula|derivative|integral|probability)\b",
            r"\b(\d+\s*[\+\-\*/]\s*\d+|\d+\s*\^\s*\d+|\d+\s*[×÷])\b"  # 简单数学表达式
        ]
        return any(re.search(pattern, q_lower, re.IGNORECASE) for pattern in math_keywords)
    
    def _is_translate_question(self, q_lower: str, original: str) -> bool:
        translate_keywords = [
            r"\b(翻译|译为|译成|英文|中文|日语|法语|德语|translate|translation)\b",
            r"\b(how to say .* in (english|chinese|japanese))\b",
            r"\b(what is .* in (chinese|english))\b"
        ]
        return any(re.search(pattern, q_lower, re.IGNORECASE) for pattern in translate_keywords)
    
    def _is_summarize_question(self, q_lower: str, original: str) -> bool:
        summarize_keywords = [
            r"\b(总结|概括|摘要|概要|简述|tl;dr|tldr|summarize|summary)\b",
            r"\b(主要观点|核心内容|大意是什么)\b",
            r"\b(in summary|to sum up|in conclusion)\b"
        ]
        return any(re.search(pattern, q_lower, re.IGNORECASE) for pattern in summarize_keywords)
    
    def _is_explanation_question(self, q_lower: str, original: str) -> bool:
        explanation_keywords = [
            r"\b(为什么|为何|怎么|如何|原理|机制|原因|explain|why|how|what is the reason)\b",
            r"\b(区别|不同|对比|比较|difference between|compare|contrast)\b",
            r"\b(详细说明|详细解释|深入分析|elaborate|describe in detail)\b"
        ]
        return any(re.search(pattern, q_lower, re.IGNORECASE) for pattern in explanation_keywords)
    
    def _is_creative_question(self, q_lower: str, original: str) -> bool:
        creative_keywords = [
            r"\b(写一篇|创作|故事|小说|诗歌|文章|essay|article|story|poem)\b",
            r"\b(想象|假如|如果|假设|imagine|suppose|what if)\b",
            r"\b(开头|结尾|情节|角色|character|plot|scene)\b"
        ]
        return any(re.search(pattern, q_lower, re.IGNORECASE) for pattern in creative_keywords)
    
    def _is_short_question(self, q_lower: str, original: str) -> bool:
        """短问题检测"""
        word_count = len(original.split())
        return word_count <= 8 and not any([
            self._is_explanation_question(q_lower, original),
            self._is_creative_question(q_lower, original)
        ])
    
    def _get_fallback_type(self, q_lower: str) -> str:
        """最终兜底逻辑"""
        word_count = len(q_lower.split())
        if word_count <= 5:
            return "qa_short"
        elif word_count <= 15:
            return "qa_explain"
        else:
            return "creative"  # 长问题倾向于创作型


# --------------------------
# 4) 输出长度预测器（增强版）
# --------------------------
class LengthPredictor:
    def __init__(self,
                 csv_path: Optional[str] = None,
                 tasks_for_prior: Optional[Dict[str, Dict[str, float]]] = None):
        self.ohe: Optional['OneHotEncoder'] = None
        self.reg_mean = None
        self.reg_q50 = None
        self.reg_q90 = None
        self.has_model = False
        self.tasks_prior = tasks_for_prior or self._default_priors()

        if csv_path and os.path.exists(csv_path) and _HAS_SK:
            try:
                df = pd.read_csv(csv_path)
                if len(df) > 10:  # 只有足够数据时才训练模型
                    self._fit_models(df)
                    self.has_model = True
                    logger.info(f"基于历史数据训练预测模型，样本数: {len(df)}")
                else:
                    logger.warning("历史数据不足，使用先验知识")
            except Exception as e:
                logger.error(f"加载历史数据失败: {e}, 使用先验知识")

    def _default_priors(self) -> Dict[str, Dict[str, float]]:
        """默认先验知识（可根据实际数据调整）"""
        return {
            "translate": {"p50_a": 1.00, "p50_b": 0,  "p90_a": 1.20, "p90_b": 10},
            "summarize": {"p50_a": 0.30, "p50_b": 30, "p90_a": 0.45, "p90_b": 50},
            "code":      {"p50_a": 0.65, "p50_b": 40, "p90_a": 0.90, "p90_b": 80},
            "math":      {"p50_a": 0.55, "p50_b": 30, "p90_a": 0.80, "p90_b": 60},
            "qa_explain":{"p50_a": 0.60, "p50_b": 40, "p90_a": 0.90, "p90_b": 80},
            "qa_short":  {"p50_a": 0.35, "p50_b": 40, "p90_a": 0.55, "p90_b": 80},
            "creative":  {"p50_a": 0.80, "p50_b": 100, "p90_a": 1.20, "p90_b": 200},
        }

    def _fit_models(self, df: pd.DataFrame):
        """训练预测模型"""
        try:
            if "task" not in df.columns:
                df["task"] = "qa_short"
            
            df = df[["S", "O", "task"]].dropna()
            if len(df) < 5:
                return
                
            S = df["S"].to_numpy().reshape(-1, 1)
            logS = np.log1p(df["S"].to_numpy()).reshape(-1, 1)
            task = df["task"].fillna("qa_short").to_numpy().reshape(-1, 1)

            # 兼容新版/旧版 scikit-learn
            kw = {"handle_unknown": "ignore"}
            try:
                self.ohe = OneHotEncoder(sparse_output=False, **kw)  # >=1.2
            except TypeError:
                self.ohe = OneHotEncoder(sparse=False, **kw)         # <1.2

            task_oh = self.ohe.fit_transform(task)
            X = np.hstack([np.ones_like(S), S, logS, task_oh])
            y = df["O"].to_numpy()

            self.reg_mean = LinearRegression()
            self.reg_q50 = QuantileRegressor(quantile=0.5, alpha=1.0, fit_intercept=False, solver="highs")
            self.reg_q90 = QuantileRegressor(quantile=0.9, alpha=0.1, fit_intercept=False, solver="highs")

            self.reg_mean.fit(X, y)
            self.reg_q50.fit(X, y)
            self.reg_q90.fit(X, y)
            
        except Exception as e:
            logger.error(f"训练预测模型失败: {e}")

    def _x_one(self, S: int, task: str):
        """构建单个样本特征"""
        S = max(1, min(int(S), 100000))  # 防护极端值
        base = np.array([[1.0, float(S), math.log1p(S)]])
        
        if self.ohe is not None:
            try:
                task_oh = self.ohe.transform(np.array([[task]]))
            except Exception:
                # 如果任务类型未知，使用全零
                task_oh = np.zeros((1, len(self.ohe.categories_[0])))
        else:
            task_oh = np.zeros((1, 1))
            
        return np.hstack([base, task_oh])

    def predict(self, S: int, task: str) -> Dict[str, float]:
        """预测输出长度"""
        # 输入验证
        if S <= 0:
            return {"O_mean": 10.0, "O_p50": 5.0, "O_p90": 20.0}
        
        S = max(1, min(S, 100000))  # 防止内存溢出

        if self.has_model and self.ohe is not None:
            try:
                x = self._x_one(S, task)
                o_mean = float(self.reg_mean.predict(x)[0])
                o_p50 = float(self.reg_q50.predict(x)[0])
                o_p90 = float(self.reg_q90.predict(x)[0])
            except Exception as e:
                logger.warning(f"模型预测失败，使用先验知识: {e}")
                pri = self.tasks_prior.get(task, self.tasks_prior["qa_short"])
                o_p50 = pri["p50_a"] * S + pri["p50_b"]
                o_p90 = pri["p90_a"] * S + pri["p90_b"]
                o_mean = 0.5 * (o_p50 + o_p90)
        else:
            pri = self.tasks_prior.get(task, self.tasks_prior["qa_short"])
            o_p50 = pri["p50_a"] * S + pri["p50_b"]
            o_p90 = pri["p90_a"] * S + pri["p90_b"]
            o_mean = 0.5 * (o_p50 + o_p90)

        # 合理裁剪
        o_mean = float(np.clip(o_mean, 1, 32768))
        o_p50 = float(np.clip(o_p50, 1, 32768))
        o_p90 = float(np.clip(o_p90, 1, 32768))
        
        return {"O_mean": o_mean, "O_p50": o_p50, "O_p90": o_p90}


# --------------------------
# 5) 主预测函数（带缓存和批量支持）
# --------------------------
class Q2OPredictor:
    """主预测器类"""
    
    def __init__(self,
                 history_csv: Optional[str] = None,
                 prefer_hf_model: Optional[str] = None,
                 use_chat_template: bool = False,
                 respect_len_hint: bool = True):
        
        self.token_counter = TokenCounter(
            prefer_hf_model=prefer_hf_model,
            use_chat_template=use_chat_template
        )
        self.task_detector = TaskTypeDetector()
        self.length_predictor = LengthPredictor(csv_path=history_csv)
        self.len_parser = LengthHintParser()
        self.respect_len_hint = respect_len_hint
        
        logger.info(f"Q2OPredictor初始化完成: tokenizer={self.token_counter.model_name}, "
                   f"use_chat_template={use_chat_template}")

    @lru_cache(maxsize=1000)
    def predict_single(self, question: str) -> PredictionResult:
        """预测单个问题（带缓存）"""
        # 计算输入token数
        S = self.token_counter.count(question)
        
        # 检测任务类型
        task = self.task_detector.detect(question)
        
        # 基础预测
        base_pred = self.length_predictor.predict(S, task)
        
        # 处理长度提示
        used_len_hint = False
        len_hint_value = None
        confidence = 0.6  # 基础置信度
        
        if self.respect_len_hint:
            len_hint = self.len_parser.parse_len_hint(question)
            if len_hint:
                token_estimate, unit = len_hint
                len_hint_value = token_estimate
                base_pred["O_p50"] = float(token_estimate)
                base_pred["O_p90"] = float(int(round(token_estimate * 1.2)))  # 20%余量
                base_pred["O_mean"] = 0.5 * (base_pred["O_p50"] + base_pred["O_p90"])
                used_len_hint = True
                confidence = 0.8  # 有长度提示时置信度更高
            elif self.len_parser.has_length_constraint(question):
                confidence = 0.7  # 有约束但无法解析具体数值
        
        # 创建结果对象
        return PredictionResult(
            S=S,
            task=task,
            tokenizer=self.token_counter.model_name or 
                     ("tiktoken/cl100k_base" if self.token_counter.enc else "approx"),
            O_mean=base_pred["O_mean"],
            O_p50=base_pred["O_p50"],
            O_p90=base_pred["O_p90"],
            confidence=confidence,
            used_len_hint=used_len_hint,
            len_hint_value=len_hint_value
        )

    def predict_batch(self, questions: List[str]) -> List[PredictionResult]:
        """批量预测"""
        return [self.predict_single(q) for q in questions]

    def clear_cache(self):
        """清空缓存"""
        self.predict_single.cache_clear()


# --------------------------
# 6) 便捷函数和CLI
# --------------------------
def estimate_from_question(question: str,
                           history_csv: Optional[str] = None,
                           prefer_hf_model: Optional[str] = None,
                           use_chat_template: bool = False,
                           respect_len_hint: bool = True) -> Dict[str, object]:
    """便捷函数：单次预测"""
    predictor = Q2OPredictor(
        history_csv=history_csv,
        prefer_hf_model=prefer_hf_model,
        use_chat_template=use_chat_template,
        respect_len_hint=respect_len_hint
    )
    result = predictor.predict_single(question)
    return result.__dict__


def main():
    """命令行接口"""
    ap = argparse.ArgumentParser(description="输出token数预测工具")
    ap.add_argument("--q", type=str, required=True, help="输入的问题文本")
    ap.add_argument("--hist", type=str, default="", help="可选：历史日志CSV(列: S,O,task)")
    ap.add_argument("--tok", type=str, default="", help="可选：显式指定HF tokenizer名")
    ap.add_argument("--with_chat_template", action="store_true",
                    help="按chat模板包装后计数，更贴近Chat场景")
    ap.add_argument("--no_len_hint", action="store_true",
                    help="关闭长度提示解析")
    ap.add_argument("--batch", action="store_true",
                    help="批量模式：从文件读取问题（每行一个）")
    ap.add_argument("--input_file", type=str, help="批量输入文件路径")
    ap.add_argument("--output_file", type=str, help="批量输出文件路径")
    ap.add_argument("--show_json", action="store_true", help="以JSON形式输出")
    ap.add_argument("--verbose", action="store_true", help="详细日志")

    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 批量处理模式
    if args.batch and args.input_file:
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
            
            predictor = Q2OPredictor(
                history_csv=args.hist if args.hist else None,
                prefer_hf_model=args.tok if args.tok else None,
                use_chat_template=args.with_chat_template,
                respect_len_hint=not args.no_len_hint
            )
            
            results = predictor.predict_batch(questions)
            
            output_data = []
            for q, result in zip(questions, results):
                item = {
                    "question": q,
                    **result.__dict__
                }
                output_data.append(item)
            
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                print(f"批量预测完成，结果保存至: {args.output_file}")
            else:
                for item in output_data:
                    print(json.dumps(item, ensure_ascii=False))
                    
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            return

    # 单条处理模式
    else:
        out = estimate_from_question(
            question=args.q,
            history_csv=(args.hist if args.hist else None),
            prefer_hf_model=(args.tok if args.tok else None),
            use_chat_template=args.with_chat_template,
            respect_len_hint=(not args.no_len_hint),
        )

        if args.show_json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(f"Tokenizer     : {out['tokenizer']}")
            print(f"S (tokens)    : {out['S']}")
            print(f"task type     : {out['task']}")
            print(f"O_mean        : {out['O_mean']:.1f}")
            print(f"O_p50         : {out['O_p50']:.1f}")
            print(f"O_p90         : {out['O_p90']:.1f}")
            print(f"Confidence    : {out['confidence']:.2f}")
            if out['used_len_hint']:
                print(f"Length hint   : {out['len_hint_value']} tokens")
            print(f"Used len hint : {out['used_len_hint']}")


if __name__ == "__main__":
    main()