#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1LvAsTzJWIEZsb5orG4vvJz0376mH27l2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LvAsTzJWIEZsb5orG4vvJz0376mH27l2" -O data.zip && rm -rf /tmp/cookies.txt
unzip data.zip
rm data.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1V8juN486jpeqPhKrGeuJ8WcpaCAy4D3-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1V8juN486jpeqPhKrGeuJ8WcpaCAy4D3-" -O dialoGPT.zip && rm -rf /tmp/cookies.txt
unzip dialoGPT.zip
mv dialoGPT models
rm dialoGPT.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1IzbfGOKkbbEXaoyVxBI0kD_bWODKjVpn  ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1IzbfGOKkbbEXaoyVxBI0kD_bWODKjVpn" -O discriminators.zip && rm -rf /tmp/cookies.txt
unzip discriminators.zip
mv discriminators models
rm discriminators.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rxgYyYEpWVH0qd2uxJQxC5uVIaApJMMf  ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rxgYyYEpWVH0qd2uxJQxC5uVIaApJMMf" -O scorers.zip && rm -rf /tmp/cookies.txt
unzip scorers.zip
mv scorers models
rm scorers.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gZmaQ94kQmf1-N04LmW7rN-gOrt-Sa46  ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gZmaQ94kQmf1-N04LmW7rN-gOrt-Sa46" -O evaluate.zip && rm -rf /tmp/cookies.txt
unzip evaluate.zip
mv evaluate results
rm evaluate.zip
python evaluate.py

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BEtFR694-8x61_-iANr2TMVck8QRqkA4  ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BEtFR694-8x61_-iANr2TMVck8QRqkA4" -O human_evaluation.zip && rm -rf /tmp/cookies.txt
unzip human_evaluation.zip
rm human_evaluation.zip