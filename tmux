#!/usr/bin/env bash
# SESSION="vscode`pwd | md5sum | cut -b -3`"
SESSION="HELLO"
echo $SESSION
# tmux new-session $SESSION