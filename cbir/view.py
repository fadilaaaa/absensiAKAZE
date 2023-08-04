from tkinter import *
import tkinter as tk
from tkinter import messagebox
import os
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, date
import time
import pickle
from main import face_recognition, update_dataset
from face_detect import crop_image, detect

# variable global
coords_global = []
showFrameLiveCam = True
img_global = np.array(0)
start_time = None
defaultValue = "Choose attendant"
entryNameStr = defaultValue
with open('dataname.txt', 'r') as file: # baca database nama
    data = file.readlines()
    data = [x.replace("\n", "") for x in data]
listNames = data

# apabila file attendance hari ini belum ada, maka program akan membuat
# file baru untuk menampung presensi hari ini
def create_new_attendance_file():
  csv_tmp = ["time,name,attended"] # nama kolom
  with open('dataname.txt', 'r') as file: # baca database nama
      data = file.readlines()
  
  # tulis data inisial untuk setiap nama terdaftar
  for i in range(len(data)):
    tmp = "NULL,{},False".format(data[i].replace("\n",""))
    csv_tmp.append(tmp)
  
  # buat file csv
  csv_txt = "\n".join(csv_tmp)
  with open('attendance_{}.csv'.format(date.today()), 'w') as file:
      file.write(csv_txt)
      
if not os.path.exists("attendance_{}.csv".format(date.today())):
  create_new_attendance_file()

def reset_attendance():
  with open('dataname.txt', 'r') as file: # baca database nama
    data = file.readlines()
    data = [x.replace("\n", "") for x in data]
  global listNames
  listNames = data
  
  create_new_attendance_file()
  messagebox.showinfo("Info", "Successfufly reset today attendances")

def save_new_member(entryInput):
  if entryInput.get() != "":
    with open('dataname.txt', 'r') as file: # baca database nama
      data = file.readlines()
      data = [x.replace("\n", "") for x in data]
      if entryInput.get() not in data:
        data.append(entryInput.get())
        stringVar.set(defaultValue) # remove default selection only, not the full list
        entryName['menu'].delete(0, 'end')
        for val in data:
            entryName['menu'].add_command(label=val, command=tk._setit(stringVar, val))
      
        global listNames
        listNames = data
        
        with open('dataname.txt', 'w') as fileW: # baca database nama
          newStr = "\n".join(data)
          fileW.write(newStr)
        
        messagebox.showinfo("Info", "success add new member")
      else:
        messagebox.showerror("Error", "failed to add new member")
  
def see_members():
  new_attendance = Toplevel(root)
  new_attendance.title("Create new attendance")
  
  bigFrameAttSeeMembers = Frame(new_attendance, bg="white", padx=10, pady=10)
  bigFrameAttSeeMembers.grid(column=0, row=0)
  tableFormSeeMembers=Frame(bigFrameAttSeeMembers, bg="black")
  tableFormSeeMembers.grid(column=0, row=0)
  
  with open('dataname.txt', 'r') as file: # baca database nama
      data = file.readlines()
      data = [x.replace("\n", "") for x in data]
  
  Label(tableFormSeeMembers, text='No', padx=5, pady=2).grid(row=0, column=0, padx=1, pady=1, sticky="nsew")
  Label(tableFormSeeMembers, text='Name', padx=5, pady=2).grid(row=0, column=1, padx=1, pady=1, sticky="nsew")
  
  # tampilkan isi
  for idx in range(len(data)):
      Label(tableFormSeeMembers, text=idx+1, padx=5, pady=2).grid(row=idx+1, column=0, padx=1, pady=1, sticky="nsew")
      Label(tableFormSeeMembers, text=data[idx], padx=5, pady=2).grid(row=idx+1, column=1, padx=1, pady=1, sticky="nsew")

def add_new_attendance():
  new_attendance = Toplevel(root)
  new_attendance.title("Create new attendance")
  bigFrameAttNewAtt = Frame(new_attendance, padx=10, pady=10)
  bigFrameAttNewAtt.grid(column=0, row=0)
  
  Label(bigFrameAttNewAtt, text="Register New Member", font=("Arial", 14, "bold")).grid(column=0, row=0)
  formNewAtt = Frame(bigFrameAttNewAtt, bg="grey", padx=10, pady=10)
  formNewAtt.grid(column=0, row=2)
  Label(formNewAtt, text="Name", padx=2).grid(column=0, row=1, padx=5)
  entryInput = Entry(formNewAtt)
  entryInput.grid(column=2, row=1)
  
  btnSaveNewMember = Button(formNewAtt, text="Save", command=lambda: save_new_member(entryInput))
  btnSaveNewMember.grid(column=0, row=2, padx=6, pady=5, sticky="nsew")
  

def callbackOptionMenuName(entryName):
  global entryNameStr
  entryNameStr = entryName
  
# save_file menyimpan gambar hasil deteksi yang di-crop dan resize
# file disimpan pada direktori dataset_train
def save_file(entryNameStr):
  # cek apakah wajah terdektsi dan entry nama diisikan
  if len(coords_global) != 0 and img_global and entryNameStr != defaultValue:    
      imgSnap = crop_image(img_global, coords_global, 20)
      imgSnap = imgSnap.resize((256,256))
      i = 1
      fileName = "{}-{}.jpg".format(entryNameStr,1)
      
      # looping hingga ditemukan angka terakhir pada nama yang dicari
      while os.path.exists("dataset_train/{}".format(fileName)):
        i+=1
        fileName = "{}-{}.jpg".format(entryNameStr, i)
      
      imgSnap.save("dataset_train/{}".format(fileName))
      messagebox.showinfo("Info", "success saving photo")
  else:
    messagebox.showerror("Failed", "failed saving photo")

# update_system update dataset pada direktori dataset_train
def update_system():
  global showFrameLiveCam
  showFrameLiveCam = False
  update_dataset('dataset_train')
  messagebox.showinfo("Info", "Update dataset done")
  showFrameLiveCam = True
  
# see_attendance menampilkan laporan kehadiran hari ini
def see_attendance():
  attendanceScreen = Toplevel(root)
  attendanceScreen.title("Attendance - {}".format(date.today()))

  df = pd.read_csv("attendance_{}.csv".format(date.today()), header=0)
  bigFrameAtt = Frame(attendanceScreen, bg="white", padx=10, pady=10)
  bigFrameAtt.grid(column=0, row=0)
  frameAtt = Frame(bigFrameAtt, bg="black")
  frameAtt.grid(row=0, column=0)

  # tampilkan header
  Label(frameAtt, text='Time', padx=5, pady=2).grid(row=0, column=1, padx=1, pady=1, sticky="nsew")
  Label(frameAtt, text='Name', padx=5, pady=2).grid(row=0, column=2, padx=1, pady=1, sticky="nsew")
  Label(frameAtt, text='Attended', padx=5, pady=2).grid(row=0, column=3, padx=1, pady=1, sticky="nsew")
  
  # tampilkan isi
  for idx, row in df.iterrows():
      Label(frameAtt, text=row['time'], padx=5, pady=2).grid(row=idx+1, column=1, padx=1, pady=1, sticky="nsew")
      Label(frameAtt, text=row['name'], padx=5, pady=2).grid(row=idx+1, column=2, padx=1, pady=1, sticky="nsew")
      Label(frameAtt, text="Yes" if row['attended'] else "No", padx=5, pady=2).grid(row=idx+1, column=3, padx=1, pady=1, sticky="nsew")

# haarcascade initialization
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier(haar_model)

# create main window
root = Tk()
root.bind('<Escape>', lambda e: root.quit())
root.geometry("718x392")
root.resizable(width=False, height=False)
root.title("Attendance System")
root.config(bg="skyblue")

# create left frame (menu and toolbar) 
menuFrame = Frame(root, width=200, height=600, padx=2, pady=2)
menuFrame.grid(column=0, row=0, sticky="nsew")
toolbar = Frame(menuFrame, width=150, height=500, bg="darkgrey")
toolbar.grid(row=2, column=0, padx=5, pady=2, rowspan=2)

# create middle frame (image processing and registration)
infoFrame = Frame(root, width=400, height=600, bg="white", padx=2, pady=2)
infoFrame.grid(column=1, row=0, sticky="nsew")
submitFormFrame = Frame(infoFrame, highlightbackground="black", highlightthickness=2, padx=2, pady=5)
submitFormFrame.grid(column=0, row=0, sticky="nsew")

# create right frame (live webcam and prediction)
liveCamFrame = Frame(root, width=400, height=600,padx=2, pady=2)
liveCamFrame.grid(column=2, row=0)

# create element of middle frame (form)
Label(submitFormFrame, text="Add new photo", font=("Arial", 16, "bold")).grid(column=0, row=0)
formFrame = Frame(submitFormFrame ,padx=2, pady=2)
formFrame.grid(column=0, row=1)
Label(formFrame, text="Name").grid(column=0, row=0)
stringVar = StringVar()
stringVar.set(defaultValue)
entryName = OptionMenu(formFrame, stringVar, defaultValue, *listNames, command=callbackOptionMenuName)
entryName.config(width=20)
entryName.grid(column=2, row=0, padx=2, pady=2, sticky="nsew")
btnSave = Button(submitFormFrame, text="Snap and save", command=lambda: save_file(entryNameStr))
btnSave.grid(column=0, row=2, padx=6, sticky="nsew")

# Create a Label to capture the Video frames in right frame
Label(liveCamFrame, text="Live WebCam", font=("Arial", 14, "bold")).grid(column=0, row=0, pady=2)
labelLiveCam = Label(liveCamFrame)
labelLiveCam.grid(column=0, row=4, sticky=W, pady=2)
cap = cv2.VideoCapture(0)
labelName = Label(liveCamFrame, text="person in frame:")
labelName.grid(column=0, row=2, sticky=W, pady=2)
labelName = Label(liveCamFrame)
labelName.grid(column=0, row=3, sticky=W, pady=2)
labelSnapshot = Label(infoFrame)
labelSnapshot.grid(column=0, row=4, sticky=W, pady=2)

# create menu item in left frame
Label(menuFrame, text="Menu").grid(row=1, column=0, padx=2, pady=2)
btnAttendance = Button(toolbar, text="See attendance", command=see_attendance)
btnAttendance.grid(column=0, row=2, padx=5, pady=5, sticky="nsew")
btnSeeMembers = Button(toolbar, text="See members", command=see_members)
btnSeeMembers.grid(column=0, row=4, padx=5, pady=5, sticky="nsew")
btnUpdateDataset = Button(toolbar, text="Update dataset", command=update_system)
btnUpdateDataset.grid(column=0, row=6, padx=5, pady=5,sticky="nsew")
btnAddAttendance = Button(toolbar, text="Add attendant", command=add_new_attendance)
btnAddAttendance.grid(column=0, row=8, padx=5, pady=5, sticky="nsew")
btnReset = Button(toolbar, text="Reset today", bg="firebrick1", command=reset_attendance)
btnReset.grid(column=0, row=10, pady=5, padx=5, sticky="nsew")
  
# attendance marked "name" as attended on report
def attendance(name):
  today = date.today()
  filename = "attendance_{}.csv".format(today)
  df = pd.read_csv(filename, header=0)
  tmp = df.loc[df['name'] == name]
  
  if not tmp['attended'].item():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    
    tmp['time'] = current_time
    tmp['attended'] = True
    df.loc[df['name'] == name] = tmp
    df.to_csv(filename, index=False)
    messagebox.showinfo('Attendance Info', "Horay! You're attended for today")
  
  
def show_frames():
    # Get the latest frame and convert into Image
    cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    img = cv2.flip(cv2image, 1)
    coords, img = detect(img, faceCascade)
    img = Image.fromarray(img)
    
    global img_global
    global coords_global
    img_global = img
    coords_global = coords
    
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=img.resize((400,300)))
    labelLiveCam.imgtk = imgtk
    labelLiveCam.configure(image=imgtk)
    
    # if face detected
    if len(coords) != 0 :
      imgSnap = crop_image(img, coords, 20)
      imgSnap = imgSnap.resize((256,256))
      if imgSnap != None:
        imgtkSnap = ImageTk.PhotoImage(image=imgSnap)
        labelSnapshot.imgtk = imgtkSnap
        labelSnapshot.configure(image=imgtkSnap)
        pred, score = face_recognition(np.asarray(imgSnap), opt=2)
        
        # if similarity below 8 then do attend for predicted name
        if score < 6:
          global start_time
          if start_time == None :
            start_time = time.time()
          labelName.configure(text="{} ({:3f})".format(pred, score), font=("Arial", 12, "bold"))
          if time.time() - start_time > 5 :
            attendance(pred)
            start_time = None
        
    
    # Repeat after an interval to capture continiously
    labelLiveCam.after(200, show_frames)

if showFrameLiveCam:
  show_frames()
root.mainloop()
