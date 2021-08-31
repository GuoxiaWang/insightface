#!/bin/bash
while true
do
    file="/tmp/done.txt"
    if [ -f "$file" ]
    then
     rm $file
     ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
    fi
    sleep 10s
done
