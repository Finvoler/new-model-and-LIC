import sys
with open(sys.argv[1], 'rb') as f:
    content = f.read().decode('utf-8', 'ignore')
keywords = ['鍘婚噸', '绛涢€夊悗', '璁粌闆?, '鏁版嵁鍔犺浇', '鐢ㄦ埛鏁?]
for line in content.split('\n'):
    clean = line.replace('\r', '').strip()
    if any(k in clean for k in keywords):
        # take last 150 chars to avoid progress bar junk
        print(clean[-150:])
