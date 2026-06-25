import re
import time
import difflib
import pandas as pd


# =====================================================
# 0. 설정
# =====================================================

MODE = "validate"
# MODE = "generate" : 패턴 후보 파일 생성
# MODE = "validate" : 선택한 pattern_id로 전체 buffer 검증

input_path = "롤_원단어.xlsx"

ORIGINAL_COL = "원단어"
VARIANT_COL = "우회표현"

MIN_GROUP_SIZE = 100
MIN_PATTERN_COUNT = 2
ANCHOR_LEN = 1

MATCH_MODE = "typed"
# broad = op, position, anchor만 같으면 매칭
# typed = op, position, anchor, 변형 타입까지 같으면 매칭


# =====================================================
# 1. 파일 불러오기
# =====================================================

print("📂 파일 불러오는 중...")

df = pd.read_excel(input_path)

for col in [ORIGINAL_COL, VARIANT_COL]:
    if col not in df.columns:
        raise ValueError(f"{col} 컬럼이 필요합니다.")

df = df.dropna(subset=[ORIGINAL_COL, VARIANT_COL]).copy()

df = df.rename(columns={
    ORIGINAL_COL: "bw_text",
    VARIANT_COL: "variant_word"
})

df["bw_text"] = df["bw_text"].astype(str).str.strip()
df["variant_word"] = df["variant_word"].astype(str).str.strip()
df["row_id"] = range(1, len(df) + 1)

print(f"✅ 파일 로딩 완료: {len(df)} rows")


# =====================================================
# 2. 타입 함수
# =====================================================

def char_type(ch):
    if ch == "":
        return "EMPTY"
    if ch.isdigit():
        return "DIGIT"
    if re.match(r"[ㄱ-ㅎ]", ch):
        return "CONSONANT"
    if re.match(r"[ㅏ-ㅣ]", ch):
        return "VOWEL"
    if re.match(r"[가-힣]", ch):
        return "KOREAN"
    if re.match(r"[a-zA-Z]", ch):
        return "ENG"
    if re.match(r"\s", ch):
        return "SPACE"
    return "SPECIAL"


def text_type(text):
    text = str(text)

    if text == "":
        return "EMPTY"

    types = [char_type(ch) for ch in text]

    if len(set(types)) == 1:
        base_type = types[0]
    else:
        base_type = "_".join(sorted(set(types)))

    if len(text) >= 2 and len(set(text)) == 1:
        return base_type + "_REPEAT"

    return base_type


def get_position(i1, i2, original_len):
    if i1 == 0 and i2 == original_len:
        return "WHOLE"
    if i1 == 0:
        return "PREFIX"
    if i2 == original_len:
        return "SUFFIX"
    return "MIDDLE"


# =====================================================
# 3. 변형 단위 추출
# =====================================================

def extract_mutation_units(original, variant, row_id):
    original_chars = list(original)
    variant_chars = list(variant)

    sm = difflib.SequenceMatcher(None, original_chars, variant_chars)
    units = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():

        if tag == "equal":
            continue

        original_part = "".join(original_chars[i1:i2])
        variant_part = "".join(variant_chars[j1:j2])

        left_anchor = original[max(0, i1 - ANCHOR_LEN):i1]
        right_anchor = original[i2:i2 + ANCHOR_LEN]

        if left_anchor == "":
            left_anchor = "START"
        if right_anchor == "":
            right_anchor = "END"

        op = tag.upper()
        position = get_position(i1, i2, len(original_chars))

        units.append({
            "row_id": row_id,
            "bw_text": original,
            "variant_word": variant,

            "op": op,
            "position": position,

            "left_anchor": left_anchor,
            "right_anchor": right_anchor,

            "original_part": original_part,
            "variant_part": variant_part,

            "original_part_type": text_type(original_part),
            "variant_part_type": text_type(variant_part),
        })

    return units


# =====================================================
# 4. 전체 mutation unit 생성
# =====================================================

print("⚙️ 전체 데이터 변형 단위 추출 중...")

all_units = []

for idx, row in df.iterrows():
    if idx % 500 == 0:
        print(f"  - 진행: {idx}/{len(df)}")

    units = extract_mutation_units(
        original=row["bw_text"],
        variant=row["variant_word"],
        row_id=row["row_id"]
    )

    all_units.extend(units)

unit_df = pd.DataFrame(all_units)

if len(unit_df) == 0:
    raise ValueError("추출된 변형 단위가 없습니다.")

print(f"✅ 변형 단위 추출 완료: {len(unit_df)} units")


# =====================================================
# 5. 설명/예시 함수
# =====================================================

def make_examples(g, limit=10):
    return " | ".join(
        g[["bw_text", "variant_word"]]
        .drop_duplicates()
        .head(limit)
        .apply(lambda x: f"{x['bw_text']}→{x['variant_word']}", axis=1)
        .tolist()
    )


def make_pattern_desc(row):
    pos_map = {
        "PREFIX": "앞부분",
        "SUFFIX": "뒷부분",
        "MIDDLE": "중간",
        "WHOLE": "전체"
    }

    op_map = {
        "REPLACE": "치환",
        "INSERT": "삽입",
        "DELETE": "삭제"
    }

    pos = pos_map.get(row["position"], row["position"])
    op = op_map.get(row["op"], row["op"])

    left = row["left_anchor"]
    right = row["right_anchor"]

    if left == "START" and right == "END":
        anchor = "전체"
    elif left == "START":
        anchor = f"시작 ~ '{right}' 앞"
    elif right == "END":
        anchor = f"'{left}' 뒤 ~ 끝"
    else:
        anchor = f"'{left}'와 '{right}' 사이"

    return (
        f"[{row['bw_text']}] {anchor} 구간의 {pos} {op}: "
        f"{row['original_part_type']} → {row['variant_part_type']}"
    )


# =====================================================
# 6. 후보 생성 함수
# =====================================================

def generate_candidates():
    print("🔍 동일 원단어 100건 이상 그룹에서 후보 도출 중...")

    eligible_words = (
        df.groupby("bw_text")
        .size()
        .reset_index(name="cnt")
    )

    eligible_words = eligible_words[
        eligible_words["cnt"] >= MIN_GROUP_SIZE
    ]["bw_text"].tolist()

    print(f"분석 대상 원단어 수: {len(eligible_words)}")

    if len(eligible_words) == 0:
        raise ValueError("100건 이상 쌓인 원단어가 없습니다. MIN_GROUP_SIZE를 낮춰보세요.")

    source = unit_df[unit_df["bw_text"].isin(eligible_words)].copy()

    group_cols = [
        "bw_text",
        "op",
        "position",
        "left_anchor",
        "right_anchor",
        "original_part_type",
        "variant_part_type"
    ]

    candidates = (
        source
        .groupby(group_cols)
        .agg(
            hit_count=("row_id", "nunique"),
            example_values=("variant_part", lambda x: ", ".join(
                pd.Series(x).drop_duplicates().astype(str).head(10)
            ))
        )
        .reset_index()
    )

    candidates = candidates[
        candidates["hit_count"] >= MIN_PATTERN_COUNT
    ].copy()

    candidates["is_too_broad"] = (
        (candidates["position"] == "WHOLE") &
        (candidates["original_part_type"] == "KOREAN") &
        (candidates["variant_part_type"] == "KOREAN")
    )

    candidates = candidates.sort_values(
        ["is_too_broad", "hit_count"],
        ascending=[True, False]
    ).reset_index(drop=True)

    candidates["pattern_id"] = range(1, len(candidates) + 1)

    candidates["pattern_desc"] = candidates.apply(
        make_pattern_desc,
        axis=1
    )

    candidates["examples"] = candidates.apply(
        lambda r: make_examples(
            source[
                (source["bw_text"] == r["bw_text"]) &
                (source["op"] == r["op"]) &
                (source["position"] == r["position"]) &
                (source["left_anchor"] == r["left_anchor"]) &
                (source["right_anchor"] == r["right_anchor"]) &
                (source["original_part_type"] == r["original_part_type"]) &
                (source["variant_part_type"] == r["variant_part_type"])
            ]
        ),
        axis=1
    )

    return candidates


# =====================================================
# 7. matcher 함수
# =====================================================

def unit_matches_pattern(unit, pattern, mode="typed"):
    if unit["op"] != pattern["op"]:
        return False

    if unit["position"] != pattern["position"]:
        return False

    if unit["left_anchor"] != pattern["left_anchor"]:
        return False

    if unit["right_anchor"] != pattern["right_anchor"]:
        return False

    if mode == "broad":
        return True

    if mode == "typed":
        return (
            unit["original_part_type"] == pattern["original_part_type"] and
            unit["variant_part_type"] == pattern["variant_part_type"]
        )

    return False


# =====================================================
# 8. generate 모드
# =====================================================

if MODE == "generate":

    candidates = generate_candidates()

    candidate_file = f"pattern_candidates_{int(time.time())}.xlsx"

    print("💾 후보 파일 저장 중...")

    with pd.ExcelWriter(candidate_file, engine="openpyxl") as writer:
        candidates.to_excel(writer, index=False, sheet_name="pattern_candidates")
        unit_df.to_excel(writer, index=False, sheet_name="all_mutation_units")

    print(f"✅ 후보 파일 생성 완료: {candidate_file}")
    print("👉 파일 열어서 pattern_id 확인 후, MODE='validate'로 바꿔 다시 실행하세요.")


# =====================================================
# 9. validate 모드
# =====================================================

elif MODE == "validate":

    candidate_file = input("후보 파일 이름을 입력하세요: ").strip()
    selected_id = int(input("검증할 pattern_id를 입력하세요: ").strip())

    candidates = pd.read_excel(candidate_file, sheet_name="pattern_candidates")

    selected = candidates[candidates["pattern_id"] == selected_id]

    if len(selected) == 0:
        raise ValueError("해당 pattern_id가 없습니다.")

    selected = selected.iloc[0]

    print("\n✅ 선택된 패턴")
    print(selected["pattern_desc"])
    print("예시:", selected["examples"])

    print("\n🔎 선택 패턴으로 전체 buffer 검증 중...")

    matched_units = unit_df[
        unit_df.apply(
            lambda r: unit_matches_pattern(r, selected, mode=MATCH_MODE),
            axis=1
        )
    ].copy()

    matched_rows = df[
        df["row_id"].isin(matched_units["row_id"].unique())
    ].copy()

    evi_cnt = matched_rows["row_id"].nunique()

    print("\n✅ 검증 완료")
    print(f"evi_cnt = {evi_cnt}")

    key_candidate = pd.DataFrame([{
        "selected_pattern_id": selected_id,
        "key_name": f"{selected['op']}_{selected['position']}_{selected['variant_part_type']}",
        "key_value": (
            f"op={selected['op']};"
            f"position={selected['position']};"
            f"left_anchor={selected['left_anchor']};"
            f"right_anchor={selected['right_anchor']};"
            f"orig_type={selected['original_part_type']};"
            f"var_type={selected['variant_part_type']};"
            f"match_mode={MATCH_MODE}"
        ),
        "evi_cnt": evi_cnt,
        "pattern_desc": selected["pattern_desc"],
        "created_at": pd.Timestamp.now()
    }])

    candidate_evidence = matched_rows.merge(
        matched_units[[
            "row_id",
            "op",
            "position",
            "left_anchor",
            "right_anchor",
            "original_part",
            "variant_part",
            "original_part_type",
            "variant_part_type"
        ]],
        on="row_id",
        how="left"
    )

    output_file = f"pattern_validation_{selected_id}_{int(time.time())}.xlsx"

    print("💾 검증 결과 저장 중...")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        key_candidate.to_excel(writer, index=False, sheet_name="selected_key_candidate")
        candidate_evidence.to_excel(writer, index=False, sheet_name="candidate_evidence")
        matched_units.to_excel(writer, index=False, sheet_name="matched_units")

    print(f"✅ 저장 완료: {output_file}")


else:
    raise ValueError("MODE는 'generate' 또는 'validate'만 가능합니다.")