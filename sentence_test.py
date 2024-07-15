import os

from core.sentence_parser import SentenceParser, reconstruct_text
from core.utils import get_user_by_platform

sp = SentenceParser(os.path.join(os.getcwd(), "cleaned2.csv"))
df = get_user_by_platform(1, 3, 6)
key_set = list(df["key"])
print(key_set)
input()
sentences = reconstruct_text(key_set)
print()
print(sentences)
