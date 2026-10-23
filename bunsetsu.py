"""GiNZA phrase boundaries mapped to existing Whisper word timestamps."""
import math
from bisect import bisect_right


# Universal Dependencies describe syntactic relations, not certain semantic roles.
ROLES = {
    "nsubj": ("subject", "主語", "誰／何がこの動作や状態に関わるか"),
    "csubj": ("subject", "主語節", "節が主語になる"),
    "obj": ("object", "目的語", "動作が直接及ぶ対象"),
    "iobj": ("object", "間接目的語", "動作が間接的に及ぶ対象"),
    "obl": ("adverbial", "連用修飾語", "時間・場所・対象などを補う。意味は文脈で判断"),
    "advmod": ("adverbial", "副詞修飾", "動作や状態の程度・様子を説明"),
    "advcl": ("adverbial", "連用修飾節", "条件・理由・時間などを節で補う"),
    "acl": ("modifier", "連体修飾語", "節で後ろの名詞を修飾"),
    "amod": ("modifier", "形容修飾", "名詞の性質を説明"),
    "nmod": ("modifier", "名詞修飾", "名詞性成分が名詞を修飾"),
    "det": ("modifier", "連体修飾語", "後ろの名詞を限定・指示"),
    "compound": ("modifier", "複合成分", "複合表現を構成"),
    "nummod": ("modifier", "数量修飾", "数量を補う"),
    "ccomp": ("complement", "内容節", "発言・思考などの内容"),
    "xcomp": ("complement", "補文", "別の述語を補う"),
    "conj": ("parallel", "並列成分", "別の成分と並列"),
    "cc": ("parallel", "接続語", "並列成分を接続"),
    "discourse": ("other", "独立語", "談話や語気を示す"),
}


def dependency_role(root):
    if root.dep_ == "ROOT":
        predicate = root.pos_ in {"VERB", "ADJ", "AUX"} or any(t.dep_ == "cop" for t in root.children)
        return ("predicate", "述語" if predicate else "文の中心", "GiNZAが判定した文の統語的中心")
    return ROLES.get(root.dep_.split(":")[0], ("other", "その他の依存", "関係は文脈とともに判断"))


def school_component(root):
    """A teaching aid derived from UD; not a separate school-grammar parser."""
    if root is None:
        return "other", "未分類"
    dep = root.dep_.split(":")[0]
    if dep == "ROOT":
        predicate = root.pos_ in {"VERB", "ADJ", "AUX"} or any(t.dep_ == "cop" for t in root.children)
        return ("predicate", "述語") if predicate else ("other", "文の中心")
    if dep in {"nsubj", "csubj"}:
        return "subject", "主語"
    if dep in {"det", "acl", "amod", "nmod", "nummod"}:
        return "modifier", "連体修飾語"
    if dep in {"obj", "iobj", "obl", "advmod", "advcl", "ccomp", "xcomp"}:
        return "adverbial", "連用修飾語"
    if dep in {"cc", "mark"}:
        return "parallel", "接続語"
    if dep == "conj":
        return "parallel", "並列関係"
    if dep in {"discourse", "vocative"}:
        return "other", "独立語"
    return "other", "その他"


def kata_to_hira(text: str) -> str:
    """Convert Katakana to Hiragana."""
    res = []
    for c in text:
        code = ord(c)
        if 0x30A1 <= code <= 0x30F6:  # ァ to ヶ
            res.append(chr(code - 0x60))
        else:
            res.append(c)
    return "".join(res)


def text_parts(doc, left, right, head_index=None, phrase_start=0.0, phrase_end=0.0):
    """Preserve text and distinguish token POS from the bunsetsu's head with start/end timing and cleaned reading."""
    parts = []
    cursor = left
    phrase_len = max(1, right - left)
    phrase_dur = max(0.0, phrase_end - phrase_start)

    for token in doc:
        start, end = max(left, token.idx), min(right, token.idx + len(token.text))
        if start >= end:
            continue
        if cursor < start:
            s_frac = (cursor - left) / phrase_len
            e_frac = (start - left) / phrase_len
            parts.append({
                "text": doc.text[cursor:start],
                "pos": "SPACE",
                "noun": False,
                "start": round(phrase_start + s_frac * phrase_dur, 2),
                "end": round(phrase_start + e_frac * phrase_dur, 2),
            })

        raw_reading = ""
        morph_reading = token.morph.get("Reading")
        if morph_reading:
            raw_reading = morph_reading[0]
        elif hasattr(token._, "reading") and token._.reading:
            raw_reading = token._.reading

        hira_reading = kata_to_hira(raw_reading)
        surface_hira = kata_to_hira(token.text)
        # Omit reading if identical to surface text or surface is already purely kana
        if hira_reading == surface_hira or not raw_reading:
            clean_reading = ""
        else:
            clean_reading = hira_reading

        token_s_frac = (start - left) / phrase_len
        token_e_frac = (end - left) / phrase_len
        p_start = round(phrase_start + token_s_frac * phrase_dur, 2)
        p_end = round(phrase_start + token_e_frac * phrase_dur, 2)

        parts.append({
            "text": doc.text[start:end],
            "pos": token.pos_,
            "noun": token.pos_ in {"NOUN", "PROPN"},
            "verb": token.pos_ == "VERB" or token.tag_.startswith("動詞-"),
            "head": token.i == head_index,
            "sentence_root": token.dep_ == "ROOT",
            "lemma": token.lemma_,
            "reading": clean_reading,
            "start": p_start,
            "end": p_end,
        })
        cursor = end
    if cursor < right:
        s_frac = (cursor - left) / phrase_len
        parts.append({
            "text": doc.text[cursor:right],
            "pos": "SPACE",
            "noun": False,
            "start": round(phrase_start + s_frac * phrase_dur, 2),
            "end": round(phrase_end, 2),
        })
    return parts


def compact(text):
    return "".join(c for c in text if not c.isspace())


def align_phrases(segment, spans):
    """Keep all characters. Never invent timing by dividing sentence duration."""
    text = segment["text"]
    words = segment.get("words", [])
    compatible = bool(words) and not segment.get('timing_unreliable') and compact(text) == compact("".join(w["word"] for w in words))
    timed = []
    offset = 0
    for word in words:
        end = offset + len(compact(word["word"]))
        start_time, end_time = word["start"], word["end"]
        if not (math.isfinite(start_time) and math.isfinite(end_time)
                and segment["start"] <= start_time <= end_time <= segment["end"]):
            compatible = False
        timed.append((offset, end, start_time, end_time))
        offset = end
    result = []
    for start, end in spans:
        left, right = len(compact(text[:start])), len(compact(text[:end]))
        overlap = [w for w in timed if w[0] < right and w[1] > left] if compatible else []
        usable = bool(overlap) and overlap[-1][3] > overlap[0][2]
        result.append({
            "parent_id": segment["id"], "text": text[start:end],
            "start": overlap[0][2] if usable else segment["start"],
            "end": overlap[-1][3] if usable else segment["end"],
            "timing": ("word" if overlap[0][0] == left and overlap[-1][1] == right else "shared_word")
                if usable else "sentence",
        })
    return result


def chunk_segments(segments, max_bytes=35000):
    """Partition subtitle segments into chunks safe for Sudachi's 49149-byte limit."""
    if not segments:
        return []
    chunks = []
    current_chunk = []
    current_bytes = 0

    for i, seg in enumerate(segments):
        seg_bytes = len(seg.get("text", "").encode("utf-8"))
        if current_chunk and (current_bytes + seg_bytes > max_bytes):
            chunks.append(current_chunk)
            current_chunk = []
            current_bytes = 0

        current_chunk.append(seg)
        current_bytes += seg_bytes

        # If accumulated text is above 20KB, look for a clean break point
        if current_bytes >= 20000 and i + 1 < len(segments):
            text = seg.get("text", "").rstrip()
            next_seg = segments[i + 1]
            time_gap = next_seg.get("start", 0) - seg.get("end", 0)
            is_punct = text.endswith(("。", "！", "？", "!", "?", "…", "」", "）", ")"))
            is_polite = text.endswith(("ました", "ません", "です", "でした", "ます"))
            is_pause = time_gap >= 0.4

            if is_punct or is_pause or (is_polite and time_gap >= 0.2):
                chunks.append(current_chunk)
                current_chunk = []
                current_bytes = 0

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _process_segment_batch(batch, nlp, start_id, base_sentence_id):
    import ginza

    doc = nlp("".join(segment["text"] for segment in batch))
    bunsetsu = list(ginza.bunsetu_spans(doc))
    heads = list(ginza.bunsetu_head_tokens(doc))
    span_starts = [span.start_char for span in bunsetsu]
    span_heads = [next((h for h in heads if span.start <= h.i < span.end), None) for span in bunsetsu]
    sentence_ids = {token.i: base_sentence_id + index for index, sentence in enumerate(doc.sents) for token in sentence}
    output, locations, roots = [], [], []
    offset = 0
    for segment in batch:
        end = offset + len(segment["text"])
        boundaries = sorted({0, len(segment["text"]),
                             *(start - offset for start in span_starts if offset < start < end)})
        spans = list(zip(boundaries, boundaries[1:]))
        phrases = align_phrases(segment, spans)
        for phrase, (left, right) in zip(phrases, spans):
            left, right = left + offset, right + offset
            span_index = bisect_right(span_starts, left) - 1
            root = span_heads[span_index] if span_index >= 0 else None
            role, label, explanation = dependency_role(root) if root is not None else ("other", "空白", "分隔文字")
            component, component_label = school_component(root)
            is_predicate = root is not None and (root.pos_ in {"VERB", "ADJ"}
                or any(t.dep_ == "cop" for t in root.children))
            predicate_kind = ""
            if is_predicate:
                predicate_kind = "主句谓语" if root.dep_ == "ROOT" else "从句／修饰中的谓语"
            predicate_lemma = ginza.bunsetu_phrase_span(root).lemma_ if is_predicate else ""
            output.append({"id": start_id + len(output), **phrase, "role": role, "role_label": label,
                           "explanation": explanation, "dep": root.dep_ if root is not None else "",
                           "sentence_id": sentence_ids.get(root.i) if root is not None else None,
                           "head_id": None, "head_text": "",
                           "component": component, "component_label": component_label,
                           "is_predicate": is_predicate, "predicate_kind": predicate_kind,
                           "predicate_lemma": predicate_lemma,
                           "parts": text_parts(doc, left, right, root.i if root is not None else None, phrase.get("start", 0.0), phrase.get("end", 0.0))})
            locations.append(left)
            roots.append(root)
        offset = end
    for phrase, root in zip(output, roots):
        if root is not None and root.dep_ != "ROOT":
            head = bisect_right(locations, root.head.idx) - 1
            if head >= 0 and head != (phrase['id'] - start_id):
                phrase['head_id'] = start_id + head
                phrase['head_text'] = output[head]['text']
    for phrase in output:
        phrase["subject_ids"] = [p["id"] for p in output if p["head_id"] == phrase["id"]
                                 and p["dep"].split(":")[0] in {"nsubj", "csubj"}] if phrase["is_predicate"] else []
    next_sentence_id = base_sentence_id + len(list(doc.sents))
    return output, next_sentence_id


_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("ja_ginza")
        except Exception as e:
            _NLP = False
    return _NLP if _NLP is not False else None


def _fallback_split_segment(segment, start_id, base_sentence_id):
    """Rule-based Japanese bunsetsu chunker when GiNZA is unavailable."""
    text = segment.get("text", "")
    words = segment.get("words", [])
    seg_start = float(segment.get("start", 0.0))
    seg_end = float(segment.get("end", seg_start + 1.0))

    if not text:
        return [], base_sentence_id

    # If words timestamps exist from Whisper
    particles = {"は", "が", "を", "に", "で", "と", "から", "まで", "へ", "より", "も", "ね", "よ", "か", "て", "ば"}
    puncts = {"、", "。", "！", "？", " "}

    phrases = []
    if words:
        current_words = []
        for w in words:
            current_words.append(w)
            w_text = w.get("word", "").strip()
            if any(w_text.endswith(p) for p in particles | puncts):
                phrases.append(current_words)
                current_words = []
        if current_words:
            phrases.append(current_words)
    else:
        # Fallback character chunking by boundary markers
        phrases = [[{"word": text, "start": seg_start, "end": seg_end}]]

    output = []
    for chunk in phrases:
        p_text = "".join(w.get("word", "") for w in chunk).strip()
        if not p_text:
            continue
        p_start = float(chunk[0].get("start", seg_start))
        p_end = float(chunk[-1].get("end", seg_end))

        # Role heuristic
        role = "modifier"
        role_label = "連体修飾語"
        explanation = "修飾成分"
        component = "modifier"
        component_label = "修飾語"
        is_pred = False

        if p_text.endswith("は") or p_text.endswith("が"):
            role, role_label, explanation = "subject", "主語", "文の主語"
            component, component_label = "subject", "主語"
        elif p_text.endswith("を"):
            role, role_label, explanation = "object", "目的語", "動作の対象"
            component, component_label = "adverbial", "連用修飾語"
        elif any(p_text.endswith(p) for p in ["に", "で", "から", "まで", "へ", "より"]):
            role, role_label, explanation = "adverbial", "連用修飾語", "状況・対象を補足"
            component, component_label = "adverbial", "連用修飾語"
        elif any(p_text.endswith(v) for v in ["す", "る", "た", "だ", "ます", "ました", "です", "でした", "ない"]):
            role, role_label, explanation = "predicate", "述語", "文の述語"
            component, component_label = "predicate", "述語"
            is_pred = True

        parts = []
        for w in chunk:
            w_str = w.get("word", "")
            parts.append({
                "text": w_str,
                "pos": "NOUN" if not is_pred else "VERB",
                "noun": not is_pred,
                "start": round(float(w.get("start", p_start)), 2),
                "end": round(float(w.get("end", p_end)), 2)
            })

        output.append({
            "id": start_id + len(output),
            "text": p_text,
            "start": round(p_start, 2),
            "end": round(p_end, 2),
            "role": role,
            "role_label": role_label,
            "explanation": explanation,
            "dep": "root" if is_pred else "nmod",
            "sentence_id": base_sentence_id,
            "head_id": None,
            "head_text": "",
            "component": component,
            "component_label": component_label,
            "is_predicate": is_pred,
            "predicate_kind": "述語" if is_pred else "",
            "predicate_lemma": p_text if is_pred else "",
            "parts": parts,
            "subject_ids": []
        })

    return output, base_sentence_id + 1


def split_segments(segments):
    if not segments:
        return []

    nlp = get_nlp()
    if nlp is not None:
        try:
            batches = chunk_segments(segments, max_bytes=35000)
            all_output = []
            base_sentence_id = 0
            for batch in batches:
                batch_output, base_sentence_id = _process_segment_batch(
                    batch, nlp, start_id=len(all_output), base_sentence_id=base_sentence_id
                )
                all_output.extend(batch_output)
            return all_output
        except Exception as e:
            print(f"[Bunsetsu] GiNZA 运行时错误，切换至内置语法分词引擎: {e}")

    # 内置轻量分词保底引擎（绝不返回空列表）
    all_output = []
    base_sentence_id = 0
    for seg in segments:
        seg_output, base_sentence_id = _fallback_split_segment(seg, len(all_output), base_sentence_id)
        all_output.extend(seg_output)
    return all_output


