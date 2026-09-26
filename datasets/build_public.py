"""Build datasets/public_cases.json from public sources. Fixed SEED=42.
Run from OUTSIDE the repo dir (local datasets/ shadows the HF package):
  cd /tmp && ../sysone-bench/.venv/bin/python ../sysone-bench/datasets/build_public.py
"""
import json
import random
import sys
from pathlib import Path

REPOSITORY_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPOSITORY_ROOT)
from datasets import load_dataset

SEED = 42
rng = random.Random(SEED)
OUT = str(Path(REPOSITORY_ROOT) / "datasets" / "public_cases.json")

AGNEWS_Q = {"topic": {"type": "choice",
    "instructions": "Which section does this news article belong to?",
    "criteria": {"world": "international news, politics, war, diplomacy",
                 "sports": "sports events, athletes, matches, scores",
                 "business": "companies, markets, economy, finance",
                 "scitech": "science, technology, research, gadgets"}}}

EMOTION_Q = {"emotion": {"type": "choice",
    "instructions": "What is the dominant emotion expressed in this text?",
    "criteria": {"sadness": "sorrow, grief, disappointment",
                 "joy": "happiness, excitement, delight",
                 "love": "affection, fondness, caring",
                 "anger": "rage, irritation, fury",
                 "fear": "anxiety, dread, worry",
                 "surprise": "shock, amazement, disbelief"}}}

BANKING_INTENTS = ["card_arrival", "card_linking", "exchange_rate", "lost_or_stolen_card",
                   "order_physical_card", "pin_blocked", "refund_not_showing_up",
                   "request_refund", "terminate_account", "topping_up_by_card",
                   "top_up_failed", "transaction_charged_twice"]
BANKING_Q = {"intent": {"type": "choice",
    "instructions": "What does the customer want?",
    "criteria": {i: f"banking request about {i.replace('_', ' ')}" for i in BANKING_INTENTS}}}

XNLI_Q = {"relation": {"type": "choice",
    "instructions": "Given the premise, what is the relation of the hypothesis to it?",
    "criteria": {"entailment": "the hypothesis must be true given the premise",
                 "neutral": "the hypothesis may or may not be true",
                 "contradiction": "the hypothesis cannot be true given the premise"}}}

SST_Q = {"sentiment": {"type": "score",
    "instructions": "How positive is the sentiment of this movie review?",
    "criteria": ["very negative", "negative", "neutral", "positive", "very positive"]}}

MULTILINGUAL_Q = {"intent": {"type": "choice",
    "instructions": "What does the customer want in `message`?",
    "criteria": {"refund": "money returned or a duplicate charge reversed",
                 "technical_help": "a bug, outage or integration problem",
                 "billing_question": "a question about an invoice, plan or payment method",
                 "information": "general information, pricing or how-to",
                 "cancellation": "wants to cancel or downgrade",
                 "other": "none of the other options fits"}}}

MULTILINGUAL_CASES = [
    ({"message": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}, {"intent": "refund"}),
    ({"message": "मेरा ऐप बार-बार क्रैश हो रहा है, कृपया मदद करें।"}, {"intent": "technical_help"}),
    ({"message": "कृपया मेरी सदस्यता तुरंत रद्द करें।"}, {"intent": "cancellation"}),
    ({"message": "मेरे चालान में यह शुल्क क्या है?"}, {"intent": "billing_question"}),
    ({"message": "क्या आपके पास टीमों के लिए कोई ट्यूटोरियल है?"}, {"intent": "information"}),
    ({"message": "Me cobraron dos veces, quiero un reembolso por favor."}, {"intent": "refund"}),
    ({"message": "La aplicación se bloquea al exportar, necesito ayuda."}, {"intent": "technical_help"}),
    ({"message": "Cancele mi suscripción de inmediato."}, {"intent": "cancellation"}),
    ({"message": "¿Por qué mi plan se renovó a un precio mayor?"}, {"intent": "billing_question"}),
    ({"message": "¿Tienen algún tutorial para equipos nuevos?"}, {"intent": "information"}),
    ({"message": "On m'a facturé deux fois, je veux un remboursement."}, {"intent": "refund"}),
    ({"message": "L'application plante à l'export, aidez-moi s'il vous plaît."}, {"intent": "technical_help"}),
    ({"message": "Veuillez annuler mon abonnement immédiatement."}, {"intent": "cancellation"}),
    ({"message": "Pourquoi mon forfait a-t-il été renouvelé plus cher ?"}, {"intent": "billing_question"}),
    ({"message": "Avez-vous un tutoriel pour les nouvelles équipes ?"}, {"intent": "information"}),
    ({"message": "Mir wurde zweimal berechnet, ich möchte eine Rückerstattung."}, {"intent": "refund"}),
    ({"message": "Die App stürzt beim Export ab, bitte helfen Sie mir."}, {"intent": "technical_help"}),
    ({"message": "Bitte kündigen Sie mein Abonnement sofort."}, {"intent": "cancellation"}),
    ({"message": "Warum wurde mein Plan teurer verlängert?"}, {"intent": "billing_question"}),
    ({"message": "Gibt es ein Tutorial für neue Teams?"}, {"intent": "information"}),
    ({"message": "تم فرض رسوم مضاعفة عليّ، أريد استرداد المبلغ."}, {"intent": "refund"}),
    ({"message": "التطبيق يتعطل عند التصدير، أحتاج إلى مساعدة."}, {"intent": "technical_help"}),
    ({"message": "يرجى إلغاء اشتراكي فورًا."}, {"intent": "cancellation"}),
    ({"message": "لماذا تم تجديد خطتي بسعر أعلى؟"}, {"intent": "billing_question"}),
    ({"message": "هل لديكم شرح للفرق الجديدة؟"}, {"intent": "information"}),
]


def sample(ds, n):
    idx = rng.sample(range(len(ds)), n)
    return [ds[i] for i in idx]


def main():
    suites = {}
    ag = load_dataset("sh0416/ag_news", split="test")
    labels = ["world", "sports", "business", "scitech"]
    cases = [({"text": r["title"] + " - " + r["description"]}, {"topic": labels[r["label"] - 1]}) for r in sample(ag, 100)]
    suites["agnews"] = {"questions": AGNEWS_Q, "cases": cases}

    em = load_dataset("dair-ai/emotion", split="test")
    elabels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    cases = [({"text": r["text"]}, {"emotion": elabels[r["label"]]}) for r in sample(em, 100)]
    suites["emotion"] = {"questions": EMOTION_Q, "cases": cases}

    bk = load_dataset("mteb/banking77", split="test")
    by_intent = {}
    for r in bk:
        lab = r["label_text"].lower()
        if lab in BANKING_INTENTS:
            by_intent.setdefault(lab, []).append(r["text"])
    cases = []
    for intent in BANKING_INTENTS:
        for t in rng.sample(by_intent[intent], 8):
            cases.append(({"text": t}, {"intent": intent}))
    rng.shuffle(cases)
    suites["banking77_12"] = {"questions": BANKING_Q, "cases": cases}

    xn = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
    xlabels = ["entailment", "neutral", "contradiction"]
    rows = [r for r in sample(xn, 80) if r["label"] >= 0][:60]
    cases = [({"premise": r["premise"], "hypothesis": r["hypothesis"]},
              {"relation": xlabels[r["label"]]}) for r in rows if r["label"] >= 0]
    suites["mnli"] = {"questions": XNLI_Q, "cases": cases}

    sst = load_dataset("SetFit/sst5", split="test")
    cases = [({"text": r["text"]}, {"sentiment": int(r["label"])}) for r in sample(sst, 60)]
    suites["sst5"] = {"questions": SST_Q, "cases": cases}

    suites["multilingual_intent"] = {"questions": MULTILINGUAL_Q,
                                     "cases": [list(c) for c in MULTILINGUAL_CASES]}

    with open(OUT, "w") as f:
        json.dump({"seed": SEED, "suites": suites}, f)
    for k, v in suites.items():
        print(k, len(v["cases"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
