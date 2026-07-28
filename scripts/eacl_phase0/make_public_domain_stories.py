#!/usr/bin/env python3
"""Emit 10 public-domain Aesop fables as an authorized stories JSONL.

The fables themselves are ancient public domain; the English text here is an
original concise retelling released CC0 for research, so no third-party rights
are copied. Each sentence is stored as an (en, ko) pair and self-checked against
the real sentence splitter so `translations` stays aligned with `build_narrative_pack`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_narrative_pack import split_sentences

SOURCE_URL = "https://www.gutenberg.org/ebooks/21"
LICENSE = ("Public domain (Aesop, ancient); English text is an original retelling "
           "released CC0 for research. Fable source: Project Gutenberg #21 (Aesop's Fables).")

# Each story: (story_id, title, [(english_sentence, korean_reference), ...])
STORIES = [
    ("story-01", "The Fox and the Grapes", [
        ("A hungry fox saw clusters of ripe grapes hanging from a high vine.",
         "배고픈 여우가 높은 덩굴에 매달린 잘 익은 포도송이를 보았다."),
        ("He jumped again and again, but he could not reach them.",
         "여우는 몇 번이고 뛰어올랐지만 포도에 닿을 수 없었다."),
        ("At last he gave up and walked away with his nose in the air.",
         "마침내 그는 포기하고 코를 치켜든 채 걸어갔다."),
        ("Those grapes are surely sour, he told himself.",
         "저 포도는 분명 시큼할 거야, 하고 그는 혼잣말했다."),
        ("It is easy to despise what you cannot have.",
         "가질 수 없는 것을 하찮게 여기기는 쉬운 법이다."),
    ]),
    ("story-02", "The Tortoise and the Hare", [
        ("A hare mocked a tortoise for moving so slowly.",
         "토끼가 거북이더러 너무 느리게 움직인다고 비웃었다."),
        ("The tortoise calmly challenged the hare to a race.",
         "거북이는 침착하게 토끼에게 달리기 시합을 청했다."),
        ("Sure of winning, the hare lay down for a nap along the way.",
         "이길 것을 확신한 토끼는 도중에 누워 낮잠을 잤다."),
        ("The tortoise kept plodding forward without ever stopping.",
         "거북이는 한 번도 멈추지 않고 꾸준히 앞으로 나아갔다."),
        ("When the hare awoke, the tortoise had already crossed the finish line.",
         "토끼가 깨어났을 때 거북이는 이미 결승선을 통과한 뒤였다."),
        ("Slow and steady wins the race.",
         "느리더라도 꾸준한 자가 경주에서 이긴다."),
    ]),
    ("story-03", "The Lion and the Mouse", [
        ("A great lion caught a tiny mouse beneath his paw.",
         "커다란 사자가 앞발 아래 작은 생쥐를 붙잡았다."),
        ("The mouse begged for mercy and promised to help him one day.",
         "생쥐는 자비를 빌며 언젠가 그를 돕겠다고 약속했다."),
        ("Amused, the lion laughed and let the little creature go.",
         "재미있어하며 사자는 웃고 그 작은 짐승을 놓아주었다."),
        ("Later the lion was trapped in a hunter's net and roared in despair.",
         "훗날 사자는 사냥꾼의 그물에 걸려 절망에 차 울부짖었다."),
        ("The mouse ran up and gnawed through the ropes to free him.",
         "생쥐가 달려와 밧줄을 갉아 그를 풀어 주었다."),
        ("Even the small can repay the strong.",
         "작은 자도 강한 자에게 은혜를 갚을 수 있다."),
    ]),
    ("story-04", "The Ant and the Grasshopper", [
        ("All summer the grasshopper sang while the ants gathered grain.",
         "여름 내내 베짱이는 노래했고 개미들은 곡식을 모았다."),
        ("He teased them for working on such a fine day.",
         "베짱이는 이렇게 좋은 날에도 일한다며 개미들을 놀렸다."),
        ("When winter came, the fields were bare and cold.",
         "겨울이 오자 들판은 헐벗고 추웠다."),
        ("The starving grasshopper begged the ants for food.",
         "굶주린 베짱이는 개미들에게 먹을 것을 구걸했다."),
        ("You sang all summer, they said, so now you may dance.",
         "여름 내내 노래했으니 이제 춤이나 추라고 그들은 말했다."),
        ("It is wise to prepare today for the needs of tomorrow.",
         "내일의 필요를 위해 오늘 준비하는 것이 현명하다."),
    ]),
    ("story-05", "The Crow and the Pitcher", [
        ("A thirsty crow found a pitcher with a little water at the bottom.",
         "목마른 까마귀가 바닥에 물이 조금 남은 물병을 발견했다."),
        ("His beak was too short to reach the water below.",
         "그의 부리는 너무 짧아 아래의 물에 닿지 못했다."),
        ("One by one he dropped small pebbles into the pitcher.",
         "그는 작은 조약돌을 하나씩 물병에 떨어뜨렸다."),
        ("The water slowly rose until it reached the brim.",
         "물은 천천히 차올라 마침내 병 입구까지 올라왔다."),
        ("The crow drank his fill and flew away satisfied.",
         "까마귀는 실컷 마시고 만족하여 날아갔다."),
        ("Little by little, cleverness does the trick.",
         "조금씩 쌓인 지혜가 문제를 해결한다."),
    ]),
    ("story-06", "The Shepherd Boy and the Wolf", [
        ("A shepherd boy watched his flock near the village.",
         "양치기 소년이 마을 근처에서 양 떼를 지켰다."),
        ("For fun he cried out that a wolf was attacking the sheep.",
         "장난삼아 그는 늑대가 양을 습격한다고 소리쳤다."),
        ("The villagers ran to help but found no wolf at all.",
         "마을 사람들이 도우러 달려왔지만 늑대는 어디에도 없었다."),
        ("He played the same trick again and they came once more.",
         "그는 같은 장난을 또 쳤고 사람들은 다시 달려왔다."),
        ("When a real wolf finally came, no one believed his cries.",
         "진짜 늑대가 마침내 나타났을 때 아무도 그의 외침을 믿지 않았다."),
        ("A liar is not believed even when he speaks the truth.",
         "거짓말쟁이는 진실을 말해도 믿음을 얻지 못한다."),
    ]),
    ("story-07", "The Wolf and the Crane", [
        ("A wolf swallowed his food so fast that a bone stuck in his throat.",
         "늑대가 음식을 급히 삼키다가 뼈가 목에 걸렸다."),
        ("In great pain he promised a reward to anyone who could help.",
         "몹시 아파하며 그는 도와주는 이에게 상을 주겠다고 약속했다."),
        ("A crane put her long beak down his throat and pulled the bone out.",
         "두루미가 긴 부리를 그의 목에 넣어 뼈를 빼냈다."),
        ("She then asked the wolf for the reward he had promised.",
         "그러고 나서 두루미는 약속한 상을 늑대에게 청했다."),
        ("Your reward is that my jaws did not close on your head, he sneered.",
         "네 머리를 내 턱으로 물지 않은 것이 상이다, 하고 그는 비웃었다."),
        ("Expect no gratitude from the wicked.",
         "악한 자에게서 감사를 바라지 마라."),
    ]),
    ("story-08", "The North Wind and the Sun", [
        ("The north wind and the sun argued about who was stronger.",
         "북풍과 태양이 누가 더 센지를 두고 다투었다."),
        ("They agreed that the winner would be the one to strip a traveler of his coat.",
         "나그네의 외투를 벗기는 쪽이 이기는 것으로 그들은 정했다."),
        ("The wind blew hard, but the man only wrapped his coat tighter.",
         "바람이 세게 불었지만 나그네는 외투를 더 단단히 여밀 뿐이었다."),
        ("Then the sun shone warmly upon the road.",
         "그러자 태양이 길 위로 따뜻하게 내리쬐었다."),
        ("Soon the traveler grew hot and took his coat off himself.",
         "곧 나그네는 더워져 스스로 외투를 벗었다."),
        ("Gentle persuasion succeeds where force fails.",
         "힘으로 안 되는 일을 부드러운 설득이 이룬다."),
    ]),
    ("story-09", "The Goose and the Golden Eggs", [
        ("A poor farmer owned a goose that laid one golden egg each day.",
         "가난한 농부가 매일 황금알을 하나씩 낳는 거위를 가지고 있었다."),
        ("Each egg made him a little richer than before.",
         "알 하나하나가 그를 전보다 조금씩 더 부유하게 만들었다."),
        ("Impatient for more, he imagined a treasure hidden inside the bird.",
         "더 갖고 싶은 조바심에 그는 새 안에 보물이 숨어 있으리라 상상했다."),
        ("He killed the goose and cut it open at once.",
         "그는 거위를 죽여 단번에 배를 갈랐다."),
        ("Inside he found nothing but an ordinary bird.",
         "안에는 평범한 새 한 마리 말고는 아무것도 없었다."),
        ("Greed often destroys the very thing that provides.",
         "탐욕은 흔히 자신을 먹여 살리는 바로 그것을 망가뜨린다."),
    ]),
    ("story-10", "The Dog and His Reflection", [
        ("A dog crossed a bridge carrying a piece of meat in his mouth.",
         "개가 고기 한 점을 입에 물고 다리를 건넜다."),
        ("Looking down, he saw his own reflection in the water.",
         "아래를 내려다보다 그는 물에 비친 자기 모습을 보았다."),
        ("He thought it was another dog with a larger piece of meat.",
         "그는 그것이 더 큰 고기를 문 다른 개라고 여겼다."),
        ("Greedily he snapped at the reflection to seize its meat.",
         "욕심껏 그는 그 고기를 빼앗으려 물그림자를 물려 했다."),
        ("His own meat fell into the river and was swept away.",
         "그러자 자기 고기가 강에 떨어져 떠내려가 버렸다."),
        ("Grasp at a shadow and you may lose the substance.",
         "그림자를 붙잡으려다 실체를 잃을 수 있다."),
    ]),
]


def build_rows() -> list[dict]:
    rows = []
    for story_id, title, pairs in STORIES:
        en_sentences = [en for en, _ in pairs]
        ko_sentences = [ko for _, ko in pairs]
        text = " ".join(en_sentences)
        got = split_sentences(text)
        if got != en_sentences:
            raise SystemExit(
                f"{story_id}: splitter disagreement\n  expected {len(en_sentences)}: {en_sentences}\n"
                f"  got {len(got)}: {got}")
        rows.append({
            "story_id": story_id, "title": title, "text": text,
            "source_url": SOURCE_URL, "license": LICENSE,
            "translations": ko_sentences,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path, nargs="?",
                    default=Path("outputs/eacl_phase0/public_domain_stories.jsonl"))
    args = ap.parse_args()
    rows = build_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    total = sum(len(s[2]) for s in STORIES)
    print(f"wrote {len(rows)} public-domain stories ({total} sentences) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
