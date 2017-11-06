#!/usr/bin/python
# -*- coding: utf-8 -*-
import pyautogui
import os,sys
import time
import socket

host=''
mlpport=1235

pyautogui.FAILSAFE=False
pyautogui.PAUSE=0

pyautogui.moveTo(300,300)
pyautogui.click()

wps_server=socket.socket(socket.AF_INET,socket.SOCK_STREAM,0)
wps_server.bind((host,mlpport)) 
wps_server.listen(3)
stage = 0

while (True):
    print "circle"
    
    client,ipaddr=wps_server.accept()
    print ("Got a connect from %s"  %str(ipaddr))
    while (True):
        data=client.recv(1024)
        #if we can go test
        #signal gotest to open wps doc,  make stage = 0 
        if data == 'gotest':
            stage = 0
            pyautogui.press('enter')
        
        
        #signal state1 , make stage = 1
        if data == 'state1' and stage != 1:
            stage = 1
    
        #signal state2 to roll to endpage
        if data == 'state2' and stage != 2:
            stage = 2
            pyautogui.hotkey('ctrlleft', 'end')
        
        
        #signal state3 to close wps
        if data == 'state3' and stage == 2:
            stage = 3
            time.sleep(2)
            pyautogui.hotkey('altleft', 'f4')
            time.sleep(5)
    
            pyautogui.typewrite(' clear ', interval=0)
            time.sleep(1)
            pyautogui.press('enter')
            time.sleep(1)
            pyautogui.typewrite(' echo 3 > /proc/sys/vm/drop_caches ', interval=0)
            time.sleep(1)
            pyautogui.press('enter')
            time.sleep(1)
            pyautogui.typewrite(' clear ', interval=0)
            time.sleep(1)
            pyautogui.press('enter')
            time.sleep(1)
            pyautogui.typewrite(' ./wps ../doc/test.docx ', interval=0)
            time.sleep(1)
        if data == 'exit' :
            break

    client.close()



