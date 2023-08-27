import openai
from dotenv import (load_dotenv)
import os
import json
import spotipy
import pprint
import argparse


def search_songs(query: str) -> str:
    search_result = spotify.search(q=query, type="track", limit=5)
    return search_result['tracks']['items'][0]['uri']


def create_playlist(playlist_name: str):
    return spotify.user_playlist_create(
        user["id"],
        public=False,
        name=playlist_name
    )


def add_tracks(playlist_id: str, track_uris):
    spotify.playlist_add_items(
        playlist_id=playlist_id,
        items=track_uris,
        position=None)


def get_songs(prompt: str, count: int):
    messages = [
        {"role": "system",
         "content": """You are a helpful playlist generating assistant. 
                        You should generate a list of songs and their artists according to a text prompt.
                        You should return a JSON array, where each element should follow this format:
                        {"song": <song_title>, "artist": <artist_nme>}
                        """},
        {"role": "user", "content": f"""Generate a playlist of {count} songs based on this prompt: {prompt}"""}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=400
    )
    return json.loads(response.choices[0].message.content)


def generate_playlist(prompt: str, count: int, playlist_name: str) -> str:
    track_uris = []
    items = get_songs(prompt, count)

    for item in items:
        artist, song = item["artist"], item["song"]
        track_uri = search_songs(f"{song} - {artist}")
        track_uris.append(track_uri)

    playlist = create_playlist(playlist_name)
    add_tracks(playlist["id"], track_uris)
    return items


def main():
    parser = argparse.ArgumentParser(description="Create a Spotify playlist")
    parser.add_argument("description")
    parser.add_argument("--count")
    parser.add_argument("--name")
    args = parser.parse_args()

    items = generate_playlist(args.description, int(args.count), args.name)
    print(items)


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    spotify = spotipy.Spotify(
        auth_manager=spotipy.SpotifyOAuth(
            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
            redirect_uri="http://localhost:9999",
            scope="playlist-modify-private"
        )
    )
    user = spotify.current_user()
    assert user is not None
    main()
