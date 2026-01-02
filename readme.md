## OpenAI Application

### Color Palette Generator
To run application - http://localhost:5001/
`python .\color_palette_generator\app.py`

![preview.png](static%2Fcolor_palette_generator%2Fpreview.png)

### GPT-3.5-turbo Chatbot(CL)
To run application - `python .\chatbot\app.py --personality "rude and obnoxious"`

![preview.png](static%2Fchatbot%2Fpreview.png)

### GPT-3.5-turbo Code Reviewer(CL)
To run application - `python .\app.py .\sample\dfs.py`
![preview.png](static%2Fcode-reviewer%2Fpreview.png)
### Spotify Playlist Generator
```shell
python -m venv .venv
.\.venv\Scripts\activate.bat
python .\app.py "Gangsta rap song from West coast" --count 8 --name  "G-Rap"
```
![preview.png](static%2Fspotify-playlist-generator%2Fpreview.png)
### Embeddings w/ Nomic Atlas
Application to get embedding of 5000 movies, upload to Nomic Atlas and do recommendations. (`movie_embeddings_cache.pkl` differs from Mac to PC)
```shell
pip install nomic
nomic login [token]
```
To run application - `python .\app.py`
![preview-console.png](static%2Fembeddings%2Fpreview-console.png)
![preview-upload-atlas.png](static%2Fembeddings%2Fpreview-upload-atlas.png)
![preview-distance.png](static%2Fembeddings%2Fpreview-distance.png)
![preview-nomic.png](static%2Fembeddings%2Fpreview-nomic.png)

### Reddit Sentiment Analyzer 
List 5 different cities in the States that I've lived. Go to reddit api(praw) to grab monthly submission's comment and score them into .csv file.  
To run application - `python .\app.py`

#### Troubleshoot
[Fetch reddit client id and secret](https://reddit.com/prefs/apps)  
[Tiktoken 4 ChatCompletion](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)  
[How to add ipython console to pycharm](https://www.youtube.com/watch?v=6JpLmAWa6lA)
![preview.png](static%2Fsentiment-analysis%2Fpreview.png)