<!DOCTYPE html>

<html lang="en-us">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="X-UA-Compatible" content="IE=Edge">

  <title>Million Playlist Dataset - Spotify You</title>
  <link rel="stylesheet" href="http://localhost:4000/assets/css/just-the-docs.css">
  
  <script type="text/javascript" src="http://localhost:4000/assets/js/vendor/lunr.min.js"></script>
  
  <script type="text/javascript" src="http://localhost:4000/assets/js/just-the-docs.js"></script>

  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>


  <div class="page-wrap">
    <div class="side-bar">
      <a href="http://localhost:4000" class="site-title fs-6 lh-tight">Spotify You</a>
      <span class="fs-3"><button class="js-main-nav-trigger navigation-list-toggle btn btn-outline" type="button" data-text-toggle="Hide">Menu</button></span>
      <div class="navigation main-nav js-main-nav">
        <nav>
  <ul class="navigation-list">
    
    
      
        
          <li class="navigation-list-item">
            
            <a href="http://localhost:4000/" class="navigation-list-link">Home</a>
            
          </li>
        
      
    
      
        
          <li class="navigation-list-item">
            
            <a href="http://localhost:4000/background.html" class="navigation-list-link">Background</a>
            
          </li>
        
      
    
      
        
          <li class="navigation-list-item">
            
            <a href="http://localhost:4000/final_notebook/data.html" class="navigation-list-link">Data Exploration</a>
            
          </li>
        
      
    
      
        
          <li class="navigation-list-item">
            
            <a href="http://localhost:4000/final_notebook/models.html" class="navigation-list-link">Models</a>
            
          </li>
        
      
    
      
        
          <li class="navigation-list-item">
            
            <a href="http://localhost:4000/conclusions.html" class="navigation-list-link">Conclusions</a>
            
          </li>
        
      
    
  </ul>
</nav>

      </div>
      <footer role="contentinfo" class="site-footer">
        <p class="text-small text-grey-dk-000 mb-0">This site uses <a href="https://github.com/pmarsceill/just-the-docs">Just the Docs</a>, a documentation theme for Jekyll.</p>
      </footer>
    </div>
    <div class="main-content-wrap">
      <div class="page-header">
        <div class="main-content">
          
          <div class="search js-search">
            <div class="search-input-wrap">
              <input type="text" class="js-search-input search-input" placeholder="Search Spotify You" aria-label="Search Spotify You" autocomplete="off">
              <svg width="14" height="14" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg" class="search-icon"><title>Search</title><g fill-rule="nonzero"><path d="M17.332 20.735c-5.537 0-10-4.6-10-10.247 0-5.646 4.463-10.247 10-10.247 5.536 0 10 4.601 10 10.247s-4.464 10.247-10 10.247zm0-4c3.3 0 6-2.783 6-6.247 0-3.463-2.7-6.247-6-6.247s-6 2.784-6 6.247c0 3.464 2.7 6.247 6 6.247z"/><path d="M11.672 13.791L.192 25.271 3.02 28.1 14.5 16.62z"/></g></svg>
            </div>
            <div class="js-search-results search-results-wrap"></div>
          </div>
          
          
            <ul class="list-style-none text-small mt-md-1 mb-md-1 pb-4 pb-md-0 js-aux-nav aux-nav">
              
                <li class="d-inline-block my-0"><a href="//github.com/gzgracez/CS-109A-Final-Project">Spotify You on GitHub</a></li>
              
            </ul>
          
        </div>
      </div>
      <div class="main-content">
        
          
        
        <div class="page-content">
          # The Million Playlist Dataset
The Million Playlist Dataset contains 1,000,000 playlists created by
users on the Spotify platform.  It can be used by researchers interested
in exploring how to improve the music listening experience.

## What's in the Million Playlist Dataset
The MPD contains a million user-generated playlists. These playlists
were created during the period of January 2010 through October 2017.
Each playlist in the MPD contains a playlist title, the track list
(including track metadata) editing information (last edit time, 
number of playlist edits) and other miscellaneous information 
about the playlist. See the **Detailed
Description** section for more details.


## License
Usage of the Million Playlist Dataset is subject to these 
[license terms](https://recsys-challenge.spotify.com/license)

## Citing the Million Playlist Dataset

Citation information for the dataset can be found at
[recsys-challenge.spotify.com/dataset](https://recsys-challenge.spotify.com/dataset)


## Getting the dataset
The dataset is available at [recsys-challenge.spotify.com/dataset](https://recsys-challenge.spotify.com/dataset)

## Verifying your dataset
You can validate the dataset by checking the md5 hashes of the data.  From the top level directory of the MPD:
   
    % md5sum -c md5sums
  
This should print out OK for each of the 1,000 slice files in the dataset.

You can also compute a number of statistics for the dataset as follows:

    % python src/stats.py data
  
The output of this program should match what is in 'stats.txt'. Depending on how 
fast your computer is, stats.py can take 30 minutes or more to run.

## Detailed description
The Million Playlist Dataset consists of 1,000 slice files. These files have the naming convention of:

mpd.slice._STARTING\_PLAYLIST\_ID\_-\_ENDING\_PLAYLIST\_ID_.json

For example, the first 1,000 playlists in the MPD are in a file called 
`mpd.slice.0-999.json` and the last 1,000 playlists are in a file called
`mpd.slice.999000-999999.json`.

Each slice file is a JSON dictionary with two fields:
*info* and *playlists*.

### `info` Field
The info field is a dictionary that contains general information about the particular slice:

   * **slice** - the range of slices that in in this particular file - such as 0-999
   * ***version*** -  - the current version of the MPD (which should be v1)
   * ***generated_on*** - a timestamp indicating when the slice was generated.

### `playlists` field 
This is an array that typically contains 1,000 playlists. Each playlist is a dictionary that contains the following fields:


* ***pid*** - integer - playlist id - the MPD ID of this playlist. This is an integer between 0 and 999,999.
* ***name*** - string - the name of the playlist 
* ***description*** - optional string - if present, the description given to the playlist.  Note that user-provided playlist descrptions are a relatively new feature of Spotify, so most playlists do not have descriptions.
* ***modified_at*** - seconds - timestamp (in seconds since the epoch) when this playlist was last updated. Times are rounded to midnight GMT of the date when the playlist was last updated.
* ***num_artists*** - the total number of unique artists for the tracks in the playlist.
* ***num_albums*** - the number of unique albums for the tracks in the playlist
* ***num_tracks*** - the number of tracks in the playlist
* ***num_followers*** - the number of followers this playlist had at the time the MPD was created. (Note that the follower count does not including the playlist creator)
* ***num_edits*** - the number of separate editing sessions. Tracks added in a two hour window are considered to be added in a single editing session.
* ***duration_ms*** - the total duration of all the tracks in the playlist (in milliseconds)
* ***collaborative*** -  boolean - if true, the playlist is a collaborative playlist. Multiple users may contribute tracks to a collaborative playlist.
* ***tracks*** - an array of information about each track in the playlist. Each element in the array is a dictionary with the following fields:
   * ***track_name*** - the name of the track
   * ***track_uri*** - the Spotify URI of the track
   * ***album_name*** - the name of the track's album
   * ***album_uri*** - the Spotify URI of the album
   * ***artist_name*** - the name of the track's primary artist
   * ***artist_uri*** - the Spotify URI of track's primary artist
   * ***duration_ms*** - the duration of the track in milliseconds
   * ***pos*** - the position of the track in the playlist (zero-based)

Here's an example of a typical playlist entry:
  
        {
            "name": "musical",
            "collaborative": "false",
            "pid": 5,
            "modified_at": 1493424000,
            "num_albums": 7,
            "num_tracks": 12,
            "num_followers": 1,
            "num_edits": 2,
            "duration_ms": 2657366,
            "num_artists": 6,
            "tracks": [
                {
                    "pos": 0,
                    "artist_name": "Degiheugi",
                    "track_uri": "spotify:track:7vqa3sDmtEaVJ2gcvxtRID",
                    "artist_uri": "spotify:artist:3V2paBXEoZIAhfZRJmo2jL",
                    "track_name": "Finalement",
                    "album_uri": "spotify:album:2KrRMJ9z7Xjoz1Az4O6UML",
                    "duration_ms": 166264,
                    "album_name": "Dancing Chords and Fireflies"
                },
                {
                    "pos": 1,
                    "artist_name": "Degiheugi",
                    "track_uri": "spotify:track:23EOmJivOZ88WJPUbIPjh6",
                    "artist_uri": "spotify:artist:3V2paBXEoZIAhfZRJmo2jL",
                    "track_name": "Betty",
                    "album_uri": "spotify:album:3lUSlvjUoHNA8IkNTqURqd",
                    "duration_ms": 235534,
                    "album_name": "Endless Smile"
                },
                {
                    "pos": 2,
                    "artist_name": "Degiheugi",
                    "track_uri": "spotify:track:1vaffTCJxkyqeJY7zF9a55",
                    "artist_uri": "spotify:artist:3V2paBXEoZIAhfZRJmo2jL",
                    "track_name": "Some Beat in My Head",
                    "album_uri": "spotify:album:2KrRMJ9z7Xjoz1Az4O6UML",
                    "duration_ms": 268050,
                    "album_name": "Dancing Chords and Fireflies"
                },
                // 8 tracks omitted
                {
                    "pos": 11,
                    "artist_name": "Mo' Horizons",
                    "track_uri": "spotify:track:7iwx00eBzeSSSy6xfESyWN",
                    "artist_uri": "spotify:artist:3tuX54dqgS8LsGUvNzgrpP",
                    "track_name": "Fever 99\u00b0",
                    "album_uri": "spotify:album:2Fg1t2tyOSGWkVYHlFfXVf",
                    "duration_ms": 364320,
                    "album_name": "Come Touch The Sun"
                }
            ],

        }


## Tools
There are some tools that you can use with the dataset.
### stats.py
This python program will iterate through the entire MPD and display summary information about the contents of the MPD.

Usage:

    % python src/stats.py data
    
### show.py
This python program will show playlists given their ID.

Show playlist with PID 10:

    % python src/show.py 10
    
Show the first 500 and the last 500 playlists:

    % python src/show.py 0-500 999500-1000000
    
Show the raw json for 3 playlists:

    % python src/show.py --raw 10 20 30
        
Show 1000 playlists without the track details:

    % python src/show.py --compact 300-1300 
    
Show all playlists:

    % python src/show.py --compact 0-1000000

## How was the dataset built
The Million Playist Dataset is created by sampling playlists from the billions of playlists that Spotify users have created over the years.  Playlists that meet the following criteria are selected at random:

 * Created by a user that resides in the United States and is at least 13 years old
 * Was a public playlist at the time the MPD was generated
 * Contains at least 5 tracks
 * Contains no more than 250 tracks
 * Contains at least 3 unique artists
 * Contains at least 2 unique albums
 * Has no local tracks (local tracks are non-Spotify tracks that a user has on their local device)
 * Has at least one follower (not including the creator)
 * Was created after January 1, 2010 and before December 1, 2017
 * Does not have an offensive title
 * Does not have an adult-oriented title if the playlist was created by a user under 18 years of age

Additionally, some playlists have been modified as follows:

 * Potentially offensive playlist descriptions are removed
 * Tracks added on or after November 1, 2017 are removed

Playlists are sampled randomly, for the most part, but with some dithering to disguise the true distribution of playlists within Spotify. [Paper tracks](https://en.wikipedia.org/wiki/Fictitious_entry) may be added to some playlists to help us identify improper/unlicensed use of the dataset.

## Overall demographics of users contributing to the MPD

### Gender
 * Male: 45%
 * Female: 54%
 * Unspecified: 0.5%
 * Nonbinary: 0.5%

### Age
 * 18-24:  43%
 * 25-34:  31%
 * 35-44:   9%
 * 45-54:   4%
 * 55+:     3%
 * Other:  10%

### Country
 * US: 100%


## Who built the dataset
The million playlist dataset was built by the following researchers @ Spotify:

* Cedric De Boom
* Ching-Wei Chen
* Jean Garcia-Gathright
* Paul Lamere
* James McInerney
* Vidhya Murali
* Hugh Rawlinson
* Sravana Reddy
* Romain Yon


          
        </div>
      </div>
    </div>
  </div>
</html>
