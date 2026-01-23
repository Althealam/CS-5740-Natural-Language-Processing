import requests

url = "https://www.cs.cornell.edu/courses/cs4740/2025sp/hw0/rawtext.txt"
r = requests.get(url)
with open("/Users/althealam/Desktop/School/2026Spring/CS 5740-Natural Processing Language/CS-5740---Natural-Language-Processing/HW0/Data/rawtext.txt", "w", encoding="utf-8") as f:
    f.write(r.text)
