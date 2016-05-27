#!/bin/bash

# for i in '786' '780' '789' '729' '959' '724' '719' '787' '784' 'v72'
# do
# 	for (( x=0; x<10; x++ ))
# 	do
# 		python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i-$x.csv -m 25 -n 25 -t 0 > metrics-ss-cltss-$i-$x.csv &
# 		python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i-$x.csv -m 25 -n 25 -t 1 > metrics-ss-circular-$i-$x.csv &
# 		python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i-$x.csv -m 25 -n 25 -t 2 > metrics-ss-naive-$i-$x.csv &
# 		python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i-$x.csv -m 25 -n 25 -t 2 -o > metrics-labeled-lda-$i-$x.csv &
# 	done
# 	wait
# done

for i in 'mexican' 'chinese' 'german' 'argentine' 'vegan' 'szechuan' 'moroccan' 'malaysian' 'lebanese' 'irish' 'thai' 'russian'
do
	python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i.csv -m 25 -n 25 -t 0 > metrics-ss-cltss-$i.csv &
	python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i.csv -m 25 -n 25 -t 1 > metrics-ss-circular-$i.csv &
	python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i.csv -m 25 -n 25 -t 2 -o > metrics-labeled-lda-$i.csv &
done
wait

for i in 'italian' 'portuguese' 'taiwanese' 'japanese' 'caribbean' 'indian' 'ukrainian' 'colombian' 'cantonese' 'french'
do
	python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i.csv -m 25 -n 25 -t 0 > metrics-ss-cltss-$i.csv &
	python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i.csv -m 25 -n 25 -t 1 > metrics-ss-circular-$i.csv &
	python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i.csv -m 25 -n 25 -t 2 -o > metrics-labeled-lda-$i.csv &
done
wait

for i in 'filipino' 'vietnamese' 'pakistani' 'british' 'kosher' 'ethiopian' 'spanish' 'brazilian' 'mediterranean'  
do
	python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i.csv -m 25 -n 25 -t 0 > metrics-ss-cltss-$i.csv &
	python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i.csv -m 25 -n 25 -t 1 > metrics-ss-circular-$i.csv &
	python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i.csv -m 25 -n 25 -t 2 -o > metrics-labeled-lda-$i.csv &
done
wait

for i in 'peruvian' 'salvadoran' 'greek' 'korean' 'shanghainese' 'southern' 'pubs' 'afghan' 'cuban' 'venezuelan' 'seafood' 'african'
do
	python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i.csv -m 25 -n 25 -t 0 > metrics-ss-cltss-$i.csv &
	python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i.csv -m 25 -n 25 -t 1 > metrics-ss-circular-$i.csv &
	python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i.csv -m 25 -n 25 -t 2 -o > metrics-labeled-lda-$i.csv &
done
wait

# python3 ss_main.py -b simulation-background-$i.csv -f simulation-foreground-$i-$x.csv -m 25 -n 25 -s > metrics-s3-$i-$x.csv &
python3 ss_main.py -b simulation-background-786.csv -f simulation-foreground-786-0.csv -m 25 -n 25 -t 1 > metrics-ss-circular-786.csv &
python3 ss_main.py -b simulation-background-786.csv -f simulation-foreground-786-0.csv -m 25 -n 25 -t 2 -o > metrics-labeled-lda-786.csv &
