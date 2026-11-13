"""VistaQA zero-shot: Gemini 4 Argon answers and boxes the evidence, SAM 3 segments it, SAM 2.1 refines small targets.

OPENROUTER_API_KEY=... python vistaqa.py --inputs <VistaQA folder> --out predictions.jsonl --workers 8
"""
import argparse, base64, io, json, os, re, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment

PROMPT = """You answer a question about the image and point to the exact visual evidence for your answer.

Question: {question}

Work through these steps:
1. Premise check. List every object, attribute, piece of text and relation the question takes for granted. Look for each one in the image. If any of them is not actually there (for example, the object does not exist, the text is not written anywhere, or the stated relation is false), the premise does not hold.
2. If the premise holds, answer the question. Read text exactly as written. For counting, find and count every instance one by one.
3. Evidence. Evidence is what the answer refers to: the objects it identifies, describes or counts, and for text it reads, the smallest whole surface the text is written on; text written directly on an object much larger than the text, with no surface of its own, is boxed as the region of its characters. In a diagram or chart, text placed apart from what it names, such as a label, legend entry or caption, only supplies a name: when the answer names a part, region or element of the figure by reading such text, the evidence is what is named; when the question asks about the text itself, the evidence is the text. Objects the question mentions only to locate the referred objects are context, not evidence. Box each object instance separately; for a count, box every counted instance and nothing else. If the premise does not hold, or the answer is that what the question asks about is not there, give no evidence.

Reply with JSON only:
{{
  "operation": "count" if the answer is a number of instances, "read" if it reads visible text, otherwise "other",
  "premise_holds": true or false,
  "missing": "if the premise does not hold, a short noun phrase naming what is absent, else empty",
  "answer": "one short sentence that directly answers the question",
  "count": integer if the operation is count, else null,
  "evidence": [{{"label": "short noun phrase naming this object or surface", "text_region": true if the box is a region of written characters rather than an object or surface, else false, "box_2d": [ymin, xmin, ymax, xmax] normalized to 0-1000}}]
}}"""
CONFIRMED = "\n\nThe premise has already been checked and holds: set premise_holds to true, answer, and give evidence."
SAM3 = ("facebook/sam3", "3c879f39826c281e95690f02c7821c4de09afae7")
SAM2_TINY = ("facebook/sam2.1-hiera-tiny", "de431c4043854a71d8101e17995dfe596bf101a5")
SAM2_LARGE = ("facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610")
Q, SAME = 0.8, 0.7  # SAM 2 automatic mask generator defaults: mask quality floor, same-object IoU


def encode(m):
    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode()}


def decode(r):
    return mu.decode({"size": r["size"], "counts": r["counts"].encode()}).astype(bool)


def iou(a, b):
    i = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def pixel_box(b, w, h):  # Gemini [ymin, xmin, ymax, xmax] on 0-1000 -> pixel xyxy
    try:
        y0, x0, y1, x1 = (float(v) for v in b)
    except (TypeError, ValueError):
        return None
    (x0, x1), (y0, y1) = sorted((x0 * w / 1000, x1 * w / 1000)), sorted((y0 * h / 1000, y1 * h / 1000))
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
    return [x0, y0, x1, y1] if x1 - x0 >= 1 and y1 - y0 >= 1 else None


def parse(text):
    text = re.sub(r"<think>[\s\S]*?</think>", "", text or "")
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    for c in ([fenced.group(1)] if fenced else []) + [text]:
        s, e = c.find("{"), c.rfind("}")
        if s == -1 or e <= s:
            continue
        try:
            obj = json.loads(c[s:e + 1])
        except json.JSONDecodeError:
            try:
                obj = json.JSONDecoder().raw_decode(c[s:])[0]
            except json.JSONDecodeError:
                continue
        if isinstance(obj, dict):
            return obj


class Gemini:
    def __init__(self, key):
        from openai import OpenAI
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    def ask(self, prompt, url, temperature, effort, seed):
        from openai import BadRequestError
        msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": prompt}]}]
        for attempt in range(5):
            try:
                r = self.client.chat.completions.create(
                    model="google/gemini-4-argon", messages=msg, temperature=temperature, n=1, max_tokens=16384, seed=seed,
                    extra_body={"provider": {"only": ["google-vertex"], "allow_fallbacks": False},
                                "reasoning": {"effort": effort, "exclude": True}})
                return parse(r.choices[0].message.content)
            except BadRequestError:
                raise
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** (attempt + 1))


class Inst:
    def __init__(self, mask, box, score):
        self.mask, self.box, self.score = mask, box, score


class Sam:
    """SAM 3 (concept and box prompts) or SAM 2.1 (box prompts), Hugging Face transformers ports."""

    def __init__(self, name_rev):
        import torch, transformers as T
        self.torch, self.lock, (name, rev) = torch, threading.Lock(), name_rev
        self.sam3 = "sam3" in name
        if self.sam3:
            self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.bfloat16 if self.dev == "cuda" else torch.float32
            self.pcs = T.Sam3Model.from_pretrained(name, revision=rev, dtype=self.dtype).to(self.dev).eval()
            self.pcs_proc = T.Sam3Processor.from_pretrained(name, revision=rev)
            self.model = T.Sam3TrackerModel.from_pretrained(name, revision=rev, dtype=self.dtype).to(self.dev).eval()
            self.proc = T.Sam3TrackerProcessor.from_pretrained(name, revision=rev)
        else:
            self.dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            self.model = T.Sam2Model.from_pretrained(name, revision=rev).to(self.dev).eval()
            self.proc = T.Sam2Processor.from_pretrained(name, revision=rev)

    def _to(self, x):
        x = x.to(self.dev)
        if self.sam3 and "pixel_values" in x:
            x["pixel_values"] = x["pixel_values"].to(self.dtype)
        return x

    def concept(self, image, phrase):
        with self.lock, self.torch.no_grad():
            try:
                x = self._to(self.pcs_proc(images=image, text=" ".join(phrase.split()[:12]), return_tensors="pt"))
                out = self.pcs(**x)
            except ValueError:
                return []
            r = self.pcs_proc.post_process_instance_segmentation(
                out, threshold=0.4, mask_threshold=0.5, target_sizes=x.get("original_sizes").tolist())[0]
        return [Inst(m.cpu().numpy().astype(bool), [float(v) for v in b.tolist()], float(s))
                for m, b, s in zip(r["masks"], r["boxes"], r["scores"])]

    def boxes(self, image, boxes):  # one mask per box: the output with the highest predicted IoU
        if not boxes:
            return []
        with self.lock, self.torch.no_grad():
            x = self._to(self.proc(images=image, input_boxes=[boxes], return_tensors="pt"))
            out = self.model(**x, multimask_output=True)
            masks = self.proc.post_process_masks(out.pred_masks.cpu(), x["original_sizes"])[0]
        s = out.iou_scores[0].float().cpu().numpy()
        return [Inst(masks[i, int(s[i].argmax())].numpy().astype(bool), list(b), float(s[i].max())) for i, b in enumerate(boxes)]


def refine(tiny, large, image, rles, boxes):
    """Boxes covering <= 3% of the image: pixel majority of the SAM 3 mask, a SAM 2.1 full-image mask and a SAM 2.1
    mask from a 3x crop; SAM 2.1 large supplies the candidates where tiny is unsure or they disagree."""
    w, h = image.size
    if len(rles) != len(boxes) or not rles or any(r["size"] != [h, w] for r in rles):
        return rles

    def rect(b):
        bw, bh, cx, cy = b[2] - b[0], b[3] - b[1], (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if bw * bh / (w * h) > 0.03:
            return None
        cw, ch = max(64, 3 * bw), max(64, 3 * bh)
        return [max(0, int(cx - cw / 2)), max(0, int(cy - ch / 2)), min(w, int(np.ceil(cx + cw / 2))), min(h, int(np.ceil(cy + ch / 2)))]

    def crop(seg, j):
        l, t, r, b = rects[j]
        c = seg.boxes(image.crop(rects[j]), [[boxes[j][0] - l, boxes[j][1] - t, boxes[j][2] - l, boxes[j][3] - t]])
        if len(c) != 1:
            return None, 0.0
        m = np.zeros((h, w), bool)
        m[t:b, l:r] = c[0].mask
        return m, c[0].score

    def vote(j, *ms):
        v = np.stack([decode(rles[j])] + [m.astype(bool) for m in ms]).sum(0) >= 2
        if v.any():
            out[j] = encode(v)

    rects = [rect(b) for b in boxes]
    if not any(rects):
        return rles
    try:
        full = [i for s in range(0, len(boxes), 6) for i in tiny().boxes(image, boxes[s:s + 6])]
        if len(full) != len(boxes):
            return rles
        out, crops = list(rles), {}
        for j, inst in enumerate(full):
            if rects[j] is None or inst.score < Q:
                continue
            cm, sc = crop(tiny(), j)
            if cm is None:
                continue
            crops[j] = (cm, sc)
            if sc >= Q:
                vote(j, inst.mask, cm)
        unsure = []
        for j, inst in enumerate(full):
            if rects[j] is None:
                continue
            if inst.score < Q or j not in crops or crops[j][1] < Q:
                unsure.append(j)
                continue
            c = [rles[j], encode(inst.mask), encode(crops[j][0])]
            if mu.iou(c, c, [0] * 3)[[0, 0, 1], [1, 2, 2]].min() < SAME:
                unsure.append(j)
        if not unsure:
            return out
        try:
            big = [i for s in range(0, len(unsure), 2) for i in large().boxes(image, [boxes[j] for j in unsure[s:s + 2]])]
        except Exception:
            return out
        if len(big) != len(unsure):
            return out
        for j, inst in zip(unsure, big):
            try:
                cm, sc = crop(large(), j) if inst.score >= Q else (None, 0.0)
                if cm is not None and sc >= Q:
                    vote(j, inst.mask, cm)
            except Exception:
                pass
        return out
    except Exception:
        return rles


def clean(s):
    return re.sub(r"^(the|a|an|any)\s+", "", (s or "").strip().strip("."), flags=re.IGNORECASE)


def op(r):
    o = str(r.get("operation") or "").lower()
    return o if o in ("count", "read", "other") else None


def count_of(r):
    if not r:
        return None
    if not r.get("premise_holds", True):
        return 0
    c = r.get("count")
    return c if op(r) == "count" and isinstance(c, int) and not isinstance(c, bool) else None


class Pipeline:
    def __init__(self, gemini, sam3, tiny=None, large=None):
        self.gemini, self.sam3, self.models, self.lock = gemini, sam3, {"tiny": tiny, "large": large}, threading.Lock()

    def sam2(self, which):
        with self.lock:
            self.models[which] = self.models[which] or Sam(SAM2_TINY if which == "tiny" else SAM2_LARGE)
            return self.models[which]

    def concept(self, image, label):
        return self.sam3.concept(image, label) if label else []

    def predict(self, image, question):
        image = image.convert("RGB")
        s = min(1.0, 2048 / max(image.size))
        shown = image if s == 1.0 else image.resize((round(image.width * s), round(image.height * s)), Image.BICUBIC)
        buf = io.BytesIO()
        shown.save(buf, format="JPEG", quality=95)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        def ask(confirmed=False, effort="high", temperature=0.0, seed=1):
            return self.gemini.ask(PROMPT.format(question=question) + (CONFIRMED if confirmed else ""), url, temperature, effort, seed)

        main = ask() or {}
        if op(main) == "count":  # self-consistency: a strict majority count among three replies replaces the first
            replies = [main] + [r for r in (ask(temperature=1.0, seed=2), ask(temperature=1.0, seed=3)) if r is not None]
            counts = [count_of(r) for r in replies]
            tally = Counter(c for c in counts if c is not None)
            top = [c for c, n in tally.items() if n == max(tally.values(), default=0)]
            if len(top) == 1 and top[0] != counts[0]:
                main = replies[counts.index(top[0])]
        if not main.get("premise_holds", True) or (op(main) == "count" and main.get("count") == 0):  # abstain
            if main.get("premise_holds", True) and str(main.get("answer") or "").strip():
                return str(main["answer"]).strip(), []
            missing = clean(main.get("missing", "")) if not main.get("premise_holds", True) else ""
            if not missing:
                return "The object the question asks about is not present in the image.", []
            return f"There {'are' if re.search(r'[^s]s$', missing) else 'is'} no {missing} in the image.", []
        if not main.get("answer"):
            main = ask(confirmed=True) or {}
            if not main.get("answer"):
                main = ask(confirmed=True, effort="medium") or {}
        return str(main.get("answer", "")).strip(), self.masks(image, main)

    def masks(self, image, reply):
        w, h = image.size
        raw, items, text, unboxed = (reply.get("evidence") or [])[:80], [], [], []
        for ev in raw:
            if isinstance(ev, dict):
                label, box = clean(str(ev.get("label", ""))), ev.get("box_2d") or ev.get("bbox_2d")
                box = pixel_box(box, w, h) if isinstance(box, (list, tuple)) and len(box) == 4 else None
                if box:
                    items.append((label, box))
                    text.append(ev.get("text_region") is True)
                elif label and label not in unboxed:
                    unboxed.append(label)
        objs = [it for it, t in zip(items, text) if not t]
        if not items and unboxed:  # named without a box: search the names as concepts
            insts = sorted([i for l in unboxed for i in self.concept(image, l)], key=lambda i: -i.score)
            masks = [i.mask for i in (insts if op(reply) == "count" else insts[:1])]
        elif op(reply) == "count":
            labels = [l for l, _ in objs if l]
            masks = self.count_masks(image, Counter(labels).most_common(1)[0][0] if labels else "", [b for _, b in objs])
        else:
            masks = self.box_masks(image, objs) if objs else []
        if any(text):  # a region of written characters is its box
            seg = iter(masks)
            masks = [m for m in (self.fill(b, w, h) if t else next(seg, None) for (_, b), t in zip(items, text)) if m is not None]
        rles = [encode(m) for m in masks if m.any()]
        if len(items) == len(raw) and not unboxed and not any(text):
            rles = refine(lambda: self.sam2("tiny"), lambda: self.sam2("large"), image, rles, [b for _, b in items])
        return rles

    @staticmethod
    def fill(b, w, h):
        x0, y0, x1, y1 = (int(round(v)) for v in b)
        m = np.zeros((h, w), bool)
        m[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
        return m

    def box_masks(self, image, items):  # SAM 3 box mask, or the concept instance of its label that matches the box
        out, cache = [], {}
        for (label, box), inst in zip(items, self.sam3.boxes(image, [b for _, b in items])):
            if label.lower() not in cache:
                cache[label.lower()] = self.concept(image, label)
            cs = cache[label.lower()]
            best = max(cs, key=lambda c: iou(c.box, box), default=None)
            ok = best is not None and iou(best.box, box) >= 0.5 and best.score >= 0.5
            out.append(best.mask if ok else inst.mask)
        return out

    def count_masks(self, image, label, boxes):  # one mask per counted box (Hungarian match to concept instances)
        cs = self.concept(image, label)
        if not boxes:
            return []
        if not cs:
            return [i.mask for i in self.sam3.boxes(image, boxes)]
        m = np.array([[iou(b, c.box) for c in cs] for b in boxes])
        chosen = {r: cs[c].mask for r, c in zip(*linear_sum_assignment(-m)) if m[r, c] >= 0.5}
        rest = [i for i in range(len(boxes)) if i not in chosen]
        chosen.update({i: inst.mask for i, inst in zip(rest, self.sam3.boxes(image, [boxes[i] for i in rest]))})
        return [chosen[i] for i in range(len(boxes))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    pipe, lock = Pipeline(Gemini(os.environ["OPENROUTER_API_KEY"]), Sam(SAM3)), threading.Lock()
    done = {json.loads(l)["image"]["image_id"] for l in a.out.read_text().splitlines() if l.strip()} if a.out.exists() else set()

    def run(jp):
        rec = json.loads(jp.read_text())
        img = rec["image"]
        iid, w, h = int(img["image_id"]), int(img["width"]), int(img["height"])
        if iid in done:
            return
        stem = Path(img.get("file_name") or jp.name).stem  # file_name can carry another extension than the file
        path = next(p for p in [a.inputs / img.get("file_name", "")] + [a.inputs / f"{stem}{e}" for e in (".jpg", ".jpeg", ".png", ".webp")]
                    if p.is_file())
        try:
            text, masks = pipe.predict(Image.open(path), str(rec["question"]))
        except Exception as e:
            print(iid, type(e).__name__, e)
            text, masks = "", []
        masks = [r if r["size"] == [h, w] else encode(np.asarray(Image.fromarray(decode(r)).resize((w, h), Image.NEAREST), dtype=bool))
                 for r in masks]  # masks on the declared image size
        with lock, a.out.open("a") as f:
            f.write(json.dumps({"image": {"image_id": iid, "width": w, "height": h}, "prediction": {"text": text, "masks": masks}}) + "\n")
        print(iid, len(masks), text[:60], flush=True)

    with ThreadPoolExecutor(max(1, a.workers)) as ex:
        list(ex.map(run, sorted(a.inputs.glob("*.json"))))


if __name__ == "__main__":
    main()
