#!/bin/sh
# Launch the Emotion2Vec-S evaluations detached on allen, so an ssh drop cannot kill them.
# Progress goes to ~/ser_work/e2vs_eval.log; the sentinel file marks completion.
LOG=/home/ojjung/ser_work/e2vs_eval.log
rm -f "$LOG" /home/ojjung/ser_work/e2vs_eval.done
setsid nohup sh -c '
  sh /home/ojjung/ser_work/run_e2vs_eval.sh > /home/ojjung/ser_work/e2vs_eval.log 2>&1
  echo done > /home/ojjung/ser_work/e2vs_eval.done
' < /dev/null > /dev/null 2>&1 &
sleep 2
echo "launched; log: $LOG"
