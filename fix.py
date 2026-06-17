import codecs

with codecs.open('d:/比赛/AFAC_3/afac_agent.py', 'r', 'utf-8-sig') as f:
    text = f.read()

# Fix encoding & slashes
text = text.replace('\\"', '"')

with codecs.open('d:/比赛/AFAC_3/afac_agent.py', 'w', 'utf-8') as f:
    f.write(text)
