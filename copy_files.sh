#!/bin/bash

# پسوردی که می‌خواهید استفاده کنید
PASSWORD="P1755063881k"

# رنج IP که می‌خواهید اجرا کنید
START_IP=236
END_IP=250

# دامنه آدرس IP (فرض کنید 192.168.12.XXX)
IP_PREFIX="192.168.12."

# مسیر مقصد روی سرور ریموت
REMOTE_PATH="/home/pouria/celery/"

# نام کاربری برای اتصال به سرور ریموت
USERNAME="pouria"

# حلقه برای رنج IP
for i in $(seq $START_IP $END_IP); do
    FULL_IP="${IP_PREFIX}${i}"
    echo "Processing IP: ${FULL_IP}"
    
    # استفاده از sshpass برای ارسال پسورد و اجرای scp
    sshpass -p $PASSWORD scp *.* ${USERNAME}@${FULL_IP}:${REMOTE_PATH}
    
    if [ $? -eq 0 ]; then
        echo "Files successfully copied to ${FULL_IP}"
    else
        echo "Failed to copy files to ${FULL_IP}"
    fi

done

echo "Done processing IPs from ${START_IP} to ${END_IP}"
