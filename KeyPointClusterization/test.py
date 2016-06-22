import pickle
import zlib
CacheFile = 'Cache.bin'

data = CacheFile
data = zlib.compress(data)
cache = open(CacheFile, 'wb')
pickle.dump(data,cache)
cache.close()

print('Loading cache...')
cache = open(CacheFile, 'rb')
#data = zlib.decompress(data)
data = pickle.load(cache)
data = zlib.decompress(data)
cache.close()
strin = data
print(data)