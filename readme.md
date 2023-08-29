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
