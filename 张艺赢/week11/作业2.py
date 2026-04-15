
import random
import string
from datetime import datetime, timedelta
from collections import Counter

@mcp.tool
def analyze_text(
    text: Annotated[str, "需要分析的文本内容"],
    top_keywords: Annotated[int, "返回高频关键词的数量，默认为5"] = 5
):
    """分析文本内容，统计字数、字符数，并提取高频关键词。"""
    try:
        char_count = len(text)
        word_count = len(text.split())
        
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        english_chars = sum(1 for c in text if c.isalpha() and c.isascii())
        number_count = sum(1 for c in text if c.isdigit())
        
        words = []
        for word in text.split():
            if len(word) >= 2:
                words.append(word)
        
        keyword_freq = Counter(words)
        top_keywords_list = keyword_freq.most_common(top_keywords)
        
        return {
            "总字符数": char_count,
            "总词数": word_count,
            "中文字符数": chinese_chars,
            "英文字符数": english_chars,
            "数字字符数": number_count,
            "高频关键词": [{"关键词": k, "出现次数": v} for k, v in top_keywords_list]
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def generate_password(
    length: Annotated[int, "密码长度，建议8-32位"],
    include_uppercase: Annotated[bool, "是否包含大写字母"] = True,
    include_lowercase: Annotated[bool, "是否包含小写字母"] = True,
    include_numbers: Annotated[bool, "是否包含数字"] = True,
    include_special: Annotated[bool, "是否包含特殊字符(!@#$%^&*)"] = True
):
    """生成符合安全要求的随机密码，可自定义密码复杂度。"""
    try:
        if length < 4:
            return {"error": "密码长度至少为4位"}
        if length > 64:
            return {"error": "密码长度不能超过64位"}
        
        char_pool = ""
        required_chars = []
        
        if include_uppercase:
            char_pool += string.ascii_uppercase
            required_chars.append(random.choice(string.ascii_uppercase))
        if include_lowercase:
            char_pool += string.ascii_lowercase
            required_chars.append(random.choice(string.ascii_lowercase))
        if include_numbers:
            char_pool += string.digits
            required_chars.append(random.choice(string.digits))
        if include_special:
            special_chars = "!@#$%^&*"
            char_pool += special_chars
            required_chars.append(random.choice(special_chars))
        
        if not char_pool:
            return {"error": "至少需要选择一种字符类型"}
        
        remaining_length = length - len(required_chars)
        if remaining_length > 0:
            password_chars = required_chars + [random.choice(char_pool) for _ in range(remaining_length)]
        else:
            password_chars = required_chars[:length]
        
        random.shuffle(password_chars)
        password = ''.join(password_chars)
        
        strength = "弱"
        if length >= 12 and sum([include_uppercase, include_lowercase, include_numbers, include_special]) >= 3:
            strength = "强"
        elif length >= 8 and sum([include_uppercase, include_lowercase, include_numbers, include_special]) >= 2:
            strength = "中"
        
        return {
            "生成的密码": password,
            "密码长度": length,
            "安全强度": strength,
            "包含大写字母": include_uppercase,
            "包含小写字母": include_lowercase,
            "包含数字": include_numbers,
            "包含特殊字符": include_special
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def calculate_date(
    operation: Annotated[str, "操作类型：'add' (日期加减), 'diff' (计算日期差), 'weekday' (计算星期几)"],
    date1: Annotated[str, "第一个日期，格式为YYYY-MM-DD，默认为今天"] = None,
    date2: Annotated[str, "第二个日期，格式为YYYY-MM-DD，用于计算日期差"] = None,
    days: Annotated[int, "要加减的天数"] = 0
):
    """日期计算器，支持日期加减、计算日期间隔、查询星期几。"""
    try:
        if date1 is None:
            date1_obj = datetime.now()
        else:
            date1_obj = datetime.strptime(date1, "%Y-%m-%d")
        
        if operation == "weekday":
            weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
            return {
                "日期": date1_obj.strftime("%Y-%m-%d"),
                "星期": weekdays[date1_obj.weekday()]
            }
        
        elif operation == "add":
            result_date = date1_obj + timedelta(days=days)
            return {
                "原始日期": date1_obj.strftime("%Y-%m-%d"),
                "操作": f"增加{days}天" if days >= 0 else f"减少{abs(days)}天",
                "结果日期": result_date.strftime("%Y-%m-%d")
            }
        
        elif operation == "diff":
            if date2 is None:
                return {"error": "计算日期差需要提供date2参数"}
            date2_obj = datetime.strptime(date2, "%Y-%m-%d")
            diff_days = abs((date2_obj - date1_obj).days)
            return {
                "日期1": date1_obj.strftime("%Y-%m-%d"),
                "日期2": date2_obj.strftime("%Y-%m-%d"),
                "相差天数": diff_days,
                "相差周数": round(diff_days / 7, 1),
                "相差月数(约)": round(diff_days / 30, 1)
            }
        
        else:
            return {"error": "不支持的操作类型，请使用 add、diff 或 weekday"}
    except ValueError as e:
        return {"error": f"日期格式错误，请使用YYYY-MM-DD格式: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}
