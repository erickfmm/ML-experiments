import re

def clean_tweet(tweet:str):
    #tweet = BeautifulSoup(tweet, "lxml").get_text()
    # Eliminamos la @ y su mención
    tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
    # Eliminamos los links de las URLs
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
    # Eliminamos los links de las URLs
    tweet = re.sub(r"http?://[A-Za-z0-9./]+", ' ', tweet)
    # Nos quedamos solamente con los caracteres
    tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
    # Eliminamos espacios en blanco adicionales
    tweet = re.sub(r" +", ' ', tweet)
    return tweet