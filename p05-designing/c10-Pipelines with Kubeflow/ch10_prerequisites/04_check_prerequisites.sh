#!/bin/bash

for command in kubectl aws jq envsubst;   do     
    which $command &>/dev/null && echo "$command FOUND" || echo "$command MISSING";   
done 