import re
from typing import Dict, List, Tuple
import pandas as pd
import streamlit as st

st.set_page_config(page_title="우회표현 탐지기 v2", layout="wide")
st.title("우회표현 탐지기 v2 — 유사도 기반")

# =========================================================
# 한글 분해/조합 테이블
# =========================================================
CHOSUNG_LIST = [
    "ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ",
    "ㅅ","ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"
]

JUNGSUNG_LIST = [
    "ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ",
    "ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ",
    "ㅡ","ㅢ","ㅣ"
]

JONGSUNG_LIST = [
    "","ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ",
    "ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ",
    "ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"
]

# =========================================================
# 영문 → 한글 자모 매핑
# =========================================================
ROMAN_TO_JAMO = {
    "a": "ㅏ", "e": "ㅔ", "i": "ㅣ", "o": "ㅗ", "u": "ㅜ",
    "r": "ㄹ", "l": "ㄹ",
    "n": "ㄴ", "m": "ㅁ",
    "b": "ㅂ", "p": "ㅍ",
    "s": "ㅅ", "z": "ㅈ", "j": "ㅈ",
    "k": "ㅋ", "g": "ㄱ",
    "h": "ㅎ", "t": "ㅌ", "d": "ㄷ",
    "c": "ㅊ", "w": "ㅜ", "y": "ㅣ",
}

def normalize_mixed(text: str) -> str:
    result = []
    for ch in text:
        if ch in ROMAN_TO_JAMO:
            result.append(ROMAN_TO_JAMO[ch])
        else:
            result.append(ch)
    return "".join(result)

# =========================================================
# 원단어 & 키별 대표값
# =========================================================
SEED_WORD_KEYS = {
    "지랄":    {"key1": ["ㅈㄹ"],          "key2": ["지*랄"],               "key3": ["찌랄"],                          "key4": ["짜랄"],                  "key5": ["ziral","jiral","zi랄","ji랄"],   "key6": ["ㅈi랄","지ㄹrㄹ"]},
    "씨발":    {"key1": ["ㅆㅂ"],          "key2": ["씨*발"],               "key3": ["시발"],                          "key4": ["쌰발"],                  "key5": ["ssibal","si발","씨bal"],         "key6": ["ㅆi발","씨ㅂrㄹ"]},
    "미친놈":  {"key1": ["ㅁㅊㄴ"],        "key2": ["미*친*놈"],            "key3": [],                                "key4": ["믜친놈"],                "key5": ["michin놈","미chin놈"],          "key6": ["ㅁi친놈"]},
    "병신":    {"key1": ["ㅂㅅ"],          "key2": ["병*신"],               "key3": ["뼝신"],                          "key4": ["븅신"],                  "key5": ["byeongsin","병sin"],            "key6": ["병ㅅiㄴ"]},
    "찐따":    {"key1": [],               "key2": ["찐*따"],               "key3": ["진따"],                          "key4": [],                       "key5": [],                               "key6": ["ㅉiㄴ따","찐ㄸr"]},
    "좆밥":    {"key1": [],               "key2": ["좆*밥"],               "key3": ["쫒밥"],                          "key4": ["죶밥"],                  "key5": ["jjotbab"],                      "key6": ["좆ㅂrㅂ"]},
    "좆창":    {"key1": [],               "key2": ["좆*창"],               "key3": ["쫒창"],                          "key4": ["죶창"],                  "key5": [],                               "key6": ["좆ㅊrㅇ"]},
    "창녀":    {"key1": [],               "key2": ["창*녀"],               "key3": [],                                "key4": [],                       "key5": [],                               "key6": ["ㅊrㅇ녀"]},
    "개새끼":  {"key1": ["ㄱㅅㄲ"],        "key2": ["개*새*끼"],            "key3": ["깨새끼","개쌔끼","개새기"],        "key4": ["개쉐끼"],                "key5": [],                               "key6": ["개새ㄲi"]},
    "틀딱":    {"key1": [],               "key2": ["틀*딱"],               "key3": ["틀닥"],                          "key4": [],                       "key5": [],                               "key6": ["틀ㄸrㄱ"]},
    "빡대가리":{"key1": [],               "key2": ["빡*대*가*리"],         "key3": ["박대가리","빡때가리","빡대까리"],  "key4": ["빡대가릐"],              "key5": [],                               "key6": ["ㅃrㄱ대가리","빡대가ㄹi"]},
    "개자식":  {"key1": [],               "key2": ["개*자*식"],            "key3": ["깨자식","개짜식","개자씩"],        "key4": ["개자슥"],                "key5": [],                               "key6": ["개ㅈr식","개자ㅅiㄱ"]},
    "쌍놈":    {"key1": ["ㅆㄴ"],          "key2": ["쌍*놈"],               "key3": ["상놈"],                          "key4": ["썅놈"],                  "key5": [],                               "key6": ["ㅆrㅇ놈"]},
    "염병":    {"key1": [],               "key2": ["염*병"],               "key3": ["염뼝"],                          "key4": ["옘병"],                  "key5": [],                               "key6": []},
    "좆까":    {"key1": ["ㅈㄲ"],          "key2": ["좆*까"],               "key3": ["쫒까","좆가"],                   "key4": ["즂까"],                  "key5": ["jjotkka"],                      "key6": ["좆ㄲr"]},
    "미친년":  {"key1": ["ㅁㅊㄴ"],        "key2": ["미*친*년"],            "key3": [],                                "key4": ["미칀년"],                "key5": ["michin년","미chin년"],          "key6": ["ㅁi친년"]},
    "등신":    {"key1": [],               "key2": ["등*신"],               "key3": ["뜽신","등씬"],                   "key4": ["등싄"],                  "key5": [],                               "key6": ["등ㅅiㄴ"]},
    "미친새끼":{"key1": ["ㅁㅊㅅㄲ"],      "key2": ["미*친*새*끼"],         "key3": ["미친쌔끼","미친쌔기"],            "key4": ["믜친새끼"],              "key5": [],                               "key6": ["ㅁi친새끼"]},
    "정신병자":{"key1": [],               "key2": ["정*신*병*자"],         "key3": ["쩡신병자","정씬병자","정신뼝자","정신병짜"], "key4": ["정신병쟈"], "key5": [],                               "key6": ["정ㅅiㄴ병자","정신병ㅈr"]},
    "뻐큐":    {"key1": [],               "key2": ["뻐*큐"],               "key3": ["버큐"],                          "key4": ["빠큐"],                  "key5": ["fuck"],                         "key6": []},
    "씹창":    {"key1": [],               "key2": ["씹*창"],               "key3": ["십창"],                          "key4": ["씝창"],                  "key5": [],                               "key6": ["ㅆiㅂ창","씹ㅊrㅇ"]},
    "창놈":    {"key1": [],               "key2": ["창*놈"],               "key3": [],                                "key4": ["창넘"],                  "key5": [],                               "key6": ["ㅊrㅇ놈"]},
    "똘빡":    {"key1": [],               "key2": ["똘*빡"],               "key3": ["돌빡","똘박"],                   "key4": [],                       "key5": [],                               "key6": ["똘ㅃrㄱ"]},
    "빠갈":    {"key1": [],               "key2": ["빠*갈"],               "key3": ["바갈","빠깔"],                   "key4": [],                       "key5": [],                               "key6": ["ㅃr갈"]},
    "똘추":    {"key1": [],               "key2": ["똘*추"],               "key3": ["돌추"],                          "key4": ["똘츄"],                  "key5": [],                               "key6": []},
    "느금마":  {"key1": ["ㄴㄱㅁ"],        "key2": ["느*금*마"],            "key3": ["느끔마"],                        "key4": ["늬금마"],                "key5": [],                               "key6": ["느금ㅁr"]},
    "니미럴":  {"key1": [],               "key2": ["니*미*럴"],            "key3": [],                                "key4": ["늬미럴"],                "key5": [],                               "key6": ["ㄴi미럴"]},
    "썅년":    {"key1": [],               "key2": ["썅*년"],               "key3": ["샹년"],                          "key4": ["쌍년"],                  "key5": [],                               "key6": []},
    "느개비":  {"key1": ["ㄴㄱㅂ"],        "key2": ["느*개*비"],            "key3": ["느깨비","느개삐"],               "key4": ["느게비"],                "key5": [],                               "key6": ["느개ㅂi"]},
    "좆":      {"key1": [],               "key2": [],                      "key3": ["쫒"],                            "key4": ["죶"],                    "key5": ["jjot"],                         "key6": []},
    "존나":    {"key1": ["ㅈㄴ"],          "key2": ["존*나"],               "key3": ["쫀나"],                          "key4": ["쥰나"],                  "key5": ["jonna"],                        "key6": ["존ㄴr"]},
}

SEED_WORDS = list(SEED_WORD_KEYS.keys())

# =========================================================
# 화이트리스트
# =========================================================
WHITELIST = {
    "병실","병원","병정","병사","병장","병력","병동","병자","병리",
    "존재","존중","존경","존엄","존칭",
    "창문","창고","창작","창업","창립","창원","창녕","창덕궁","창경궁",
    "개구리","개나리","개미","개울","개인","개발","개요","개선","개정","개념",
    "등신대","등신불",
    "꽃병","꽃밭","꽃길","꽃가루","꽃향기",
    "엿기름","엿장수",
    "지란지교","지란",
}

# =========================================================
# 조사/어미
# =========================================================
POSTFIXES = [
    "이었잖아","였잖아","이잖아","잖아",
    "이었다","였다","이다","이라고","라고",
    "에서는","에서","에게","에",
    "으로는","으로","로는","로",
    "이랑","랑","하고","처럼","같이",
    "까지","부터","보다","만","도",
    "입니다","이에요","예요","아요","어요",
    "했네","하네","네","요",
    "은","는","이","가","을","를","야","아","다"
]

# =========================================================
# 공통 함수
# =========================================================
def is_korean_char(ch: str) -> bool:
    return "가" <= ch <= "힣"

def is_jamo(ch: str) -> bool:
    return ch in CHOSUNG_LIST or ch in JUNGSUNG_LIST or ch in JONGSUNG_LIST

def is_pure_initial_string(text: str) -> bool:
    """
    key1 전용.
    입력이 ㄱㄴㄷ 같은 초성으로만 이루어진 경우만 True.
    부석/박수 같은 완성형 한글은 False.
    """
    cleaned = strip_noise(text)
    return bool(cleaned) and all(ch in CHOSUNG_LIST for ch in cleaned)

def decompose_char(ch: str) -> Tuple[str, str, str]:
    if not is_korean_char(ch):
        return ch, "", ""
    code = ord(ch) - ord("가")
    return (
        CHOSUNG_LIST[code // 588],
        JUNGSUNG_LIST[(code % 588) // 28],
        JONGSUNG_LIST[code % 28],
    )

def to_jamo_str(text: str) -> str:
    result = []
    for ch in text:
        if is_korean_char(ch):
            cho, jung, jong = decompose_char(ch)
            result.append(cho)
            result.append(jung)
            if jong:
                result.append(jong)
        else:
            result.append(ch)
    return "".join(result)

def strip_noise(text: str) -> str:
    return re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z]+", "", str(text))

def strip_postfix(token: str) -> str:
    for suffix in sorted(POSTFIXES, key=len, reverse=True):
        if token.endswith(suffix) and len(token) > len(suffix):
            return token[:-len(suffix)]
    return token

# =========================================================
# 편집거리 / 유사도
# =========================================================
def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i

        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j - 1], prev[j], dp[j - 1])

    return dp[n]

def similarity_score(input_str: str, repr_str: str) -> float:
    a = to_jamo_str(input_str)
    b = to_jamo_str(repr_str)

    if not a or not b:
        return 0.0

    dist = levenshtein(a, b)
    max_len = max(len(a), len(b))
    return 1.0 - dist / max_len

# =========================================================
# key2 전처리
# =========================================================
def preprocess_key2(text: str) -> str:
    return re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z]+", "", str(text)).replace("*", "")

# =========================================================
# 임계값 설정
# =========================================================
THRESHOLDS = {
    "key1": 1.0,
    "key2": 1.0,
    "key3": 0.85,
    "key4": 0.85,
    "key5": 0.85,
    "key6": 0.85,
}

KEY_LABELS = {
    "key1": "키1: 초성",
    "key2": "키2: 특수문자삽입",
    "key3": "키3: 쌍자음정규화",
    "key4": "키4: 모음변형",
    "key5": "키5: romanize",
    "key6": "키6: 한글+영문혼합",
}

# =========================================================
# 후보와 원단어 대표값 비교
# =========================================================
def match_candidate(
    candidate: str,
    seed_word: str,
    thresholds: Dict[str, float],
    len_diff_ratio: float = 0.4
) -> List[Dict]:
    candidate_clean = strip_noise(candidate)

    is_roman = bool(re.match(r"^[a-zA-Z]+$", candidate_clean))
    matched = []
    key_data = SEED_WORD_KEYS[seed_word]

    for key, repr_list in key_data.items():
        if not repr_list:
            continue

        # key1은 순수 초성 문자열만 허용
        if key == "key1" and not is_pure_initial_string(candidate_clean):
            continue

        # 길이 제한
        if candidate_clean not in repr_list:
            if key in ("key3", "key4"):
                max_len_diff = max(1, len(seed_word) * len_diff_ratio)
                if abs(len(candidate_clean) - len(seed_word)) > max_len_diff:
                    continue

        # 순수 영문 후보는 key5만 비교
        if is_roman and key != "key5":
            continue

        # 한글/혼합 후보는 key5 대표값 중 혼합형이 있을 때만 비교
        if not is_roman and key == "key5":
            has_mixed_repr = any(
                not re.match(r"^[a-zA-Z]+$", rv) for rv in repr_list
            )
            if not has_mixed_repr:
                continue

        threshold = thresholds.get(key, 0.85)
        best_score = 0.0
        best_repr = ""

        for repr_val in repr_list:
            # 대표값 완전일치
            if candidate_clean == repr_val:
                score = 1.0

            elif key == "key1":
                # 핵심 수정:
                # 초성 키는 입력이 초성 그 자체일 때만 비교.
                # 부석 -> ㅂㅅ 변환 같은 처리는 절대 하지 않음.
                score = 1.0 if candidate_clean == repr_val else 0.0

            elif key == "key2":
                repr_clean = preprocess_key2(repr_val)
                input_clean = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z]+", "", candidate)
                score = 1.0 if input_clean == repr_clean else 0.0

            elif key == "key5":
                normalized_cand = normalize_mixed(candidate_clean)
                normalized_repr = normalize_mixed(repr_val)
                score = similarity_score(normalized_cand, normalized_repr)

            else:
                normalized_cand = normalize_mixed(candidate_clean)
                normalized_repr = normalize_mixed(repr_val)
                score = similarity_score(normalized_cand, normalized_repr)

            if score > best_score:
                best_score = score
                best_repr = repr_val

        if best_score >= threshold:
            matched.append({
                "key": key,
                "label": KEY_LABELS[key],
                "score": round(best_score, 3),
                "best_repr": best_repr,
            })

    return matched

# =========================================================
# 후보 추출
# =========================================================
def sentence_to_tokens(text: str) -> List[str]:
    raw_tokens = str(text).split()
    tokens = []

    for tok in raw_tokens:
        tok = tok.strip()
        if not tok:
            continue

        tokens.append(tok)

        stripped = re.sub(r"^[^\w가-힣ㄱ-ㅎㅏ-ㅣ]+|[^\w가-힣ㄱ-ㅎㅏ-ㅣ]+$", "", tok)
        if stripped and stripped != tok:
            tokens.append(stripped)

        base = strip_postfix(stripped or tok)
        if base and base != tok:
            tokens.append(base)

    return list(dict.fromkeys([t for t in tokens if t]))

def extract_substrings(text: str, min_len: int = 2, max_len: int = 6) -> List[str]:
    cleaned = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z]+", "", str(text))
    n = len(cleaned)
    subs = []

    for i in range(n):
        for size in range(min_len, max_len + 1):
            if i + size <= n:
                subs.append(cleaned[i:i + size])

    return list(dict.fromkeys(subs))

# =========================================================
# 문장 평가
# =========================================================
def evaluate_sentence(
    text: str,
    thresholds: Dict[str, float],
    len_diff_ratio: float = 0.4
) -> Tuple[List[str], List[Dict]]:
    tokens = sentence_to_tokens(text)
    substrings = extract_substrings(text, 2, 6)
    candidates = list(dict.fromkeys(tokens + substrings))

    findings = []
    seen = set()

    for cand in candidates:
        if not cand or cand in WHITELIST:
            continue

        for seed_word in SEED_WORDS:
            matched_keys = match_candidate(cand, seed_word, thresholds, len_diff_ratio)

            if matched_keys:
                for m in matched_keys:
                    dedup = (cand, seed_word, m["key"])
                    if dedup not in seen:
                        findings.append({
                            "후보": cand,
                            "원단어": seed_word,
                            "적용키": m["label"],
                            "유사도": m["score"],
                            "대표값": m["best_repr"],
                            "판정": "차단",
                        })
                        seen.add(dedup)

    findings.sort(key=lambda x: (-x["유사도"], x["후보"], x["원단어"]))
    return candidates, findings

# =========================================================
# UI
# =========================================================
with st.expander("원단어 리스트 보기"):
    st.write(", ".join(SEED_WORDS))

st.sidebar.header("임계값 설정")
thresholds = {}

for key, default in THRESHOLDS.items():
    if key in ("key1", "key2"):
        thresholds[key] = 1.0
        st.sidebar.text(f"{KEY_LABELS[key]}: 완전일치 고정")
    else:
        thresholds[key] = st.sidebar.slider(
            KEY_LABELS[key],
            min_value=0.5,
            max_value=1.0,
            value=default,
            step=0.05,
            key=f"slider_{key}"
        )

st.sidebar.markdown("---")
st.sidebar.markdown("**유사도 기준**")
st.sidebar.markdown("1.0 = 완전일치\n\n0.85 = 기본 임계값\n\n낮을수록 더 많이 차단")

st.sidebar.markdown("---")
st.sidebar.header("길이 차이 필터")
len_diff_ratio = st.sidebar.slider(
    "길이 차이 허용 비율",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05,
    key="slider_len_diff"
)

input_text = st.text_area(
    "검사할 문장 입력",
    height=120,
    placeholder="예: ㅈㄹ하네 / zi랄 / 개새기야 / sibal"
)

if input_text:
    candidates, findings = evaluate_sentence(input_text, thresholds, len_diff_ratio)
    decision = "차단" if findings else "비차단"

    st.subheader("최종 판단")
    if decision == "차단":
        st.error("🚫 차단")
    else:
        st.success("✅ 비차단")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("추출 후보")
        st.write(", ".join(candidates) if candidates else "없음")

    with col2:
        st.subheader("감지 통계")
        if findings:
            st.metric("감지된 패턴 수", len(findings))
            st.metric("관련 원단어 수", len(set(f["원단어"] for f in findings)))
        else:
            st.metric("감지된 패턴 수", 0)
            st.metric("관련 원단어 수", 0)

    st.subheader("감지 결과")
    if findings:
        df = pd.DataFrame(findings)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("키별 감지 분포")
        key_counts = df["적용키"].value_counts().reset_index()
        key_counts.columns = ["키", "횟수"]
        st.bar_chart(key_counts.set_index("키"))
    else:
        st.write("차단 대상 없음")

    with st.expander("🔍 유사도 상세 (디버깅용)"):
        debug_rows = []

        for cand in candidates[:30]:
            if cand in WHITELIST:
                continue

            for seed_word in SEED_WORDS:
                key_data = SEED_WORD_KEYS[seed_word]

                for key, repr_list in key_data.items():
                    if not repr_list:
                        continue

                    # key1, key2는 완전일치 고정이라 디버그 유사도 제외
                    if key in ("key1", "key2"):
                        continue

                    for repr_val in repr_list:
                        normalized = normalize_mixed(strip_noise(cand))
                        normalized_repr = normalize_mixed(repr_val)
                        score = similarity_score(normalized, normalized_repr)

                        if score >= 0.5:
                            debug_rows.append({
                                "후보": cand,
                                "정규화": normalized,
                                "원단어": seed_word,
                                "키": KEY_LABELS[key],
                                "대표값": repr_val,
                                "유사도": round(score, 3),
                            })

        if debug_rows:
            debug_df = pd.DataFrame(debug_rows).sort_values("유사도", ascending=False)
            st.dataframe(debug_df, use_container_width=True, hide_index=True)
        else:
            st.write("유사도 0.5 이상 없음")
