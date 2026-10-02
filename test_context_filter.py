from src.context_filter import remove_following_articles


text = """making any special provision for the protection of women and
children.
2
[
25A.
The State shall provide free and compulsory education
to all children of the age of five to sixteen years in such manner
as may be determined by law.]
26.
(1)
In respect of access to places of public entertainment
"""


cleaned = remove_following_articles(text)


print("\n" + "=" * 100)
print("ORIGINAL")
print("=" * 100)

print(text)


print("\n" + "=" * 100)
print("CLEANED")
print("=" * 100)

print(cleaned)
