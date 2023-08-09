from lib.face import detect, crop_image
from lib.database import get_attendances, save_data
from lib.matcher import get_featureAKAZE, who_is_this
from tkinter import *
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime, date
import time

# Create a VideoCapture object
cap = cv2.VideoCapture(0)
attendances = []
choords = []
desc = None
start_time = None


def create_new_attendance_file():
    data = []
    for idx in range(len(attendances)):
        data.append(
            {'time': None, 'name': attendances[idx], 'attended': False})
    df = pd.DataFrame(data)
    df.to_csv(f'kehadiran/attendance_{date.today()}.csv', index=False)


def set_attendance(paramsset_attendance=None, reRender=False):
    global attendances
    attendances = get_attendances()
    if paramsset_attendance is not None:
        # listName = paramsset_attendance['attendances']
        global fromFrame
        fromFrame = paramsset_attendance['formFrame']
        global StringVar
        StringVar = paramsset_attendance['stringVar']
        global defaultValue
        defaultValue = paramsset_attendance['defaultValue']
        entryName = OptionMenu(fromFrame, StringVar, defaultValue,
                               *attendances, command=lambda _: print(
                                   attendances)
                               )
        entryName.config(width=20)
        entryName.grid(column=2, row=0, padx=2, pady=2, sticky="nsew")
    if reRender:
        entryName = OptionMenu(fromFrame, StringVar, defaultValue,
                               *attendances, command=lambda _: print(
                                   attendances)
                               )
        entryName.config(width=20)
        entryName.grid(column=2, row=0, padx=2, pady=2, sticky="nsew")
    if not os.path.exists("kehadiran/attendance_{}.csv".format(date.today())):
        create_new_attendance_file()


def save_new_member(entryInput, window):
    name = entryInput.get()
    if name == "":
        messagebox.showwarning("Warning", "Please enter a name")
    else:
        save_data(name)
        set_attendance(reRender=True)
        window.destroy()
        messagebox.showinfo("Success", "New member added successfully")


def add_new_member(rootTK):
    new_attendance = Toplevel(rootTK)
    new_attendance.title("Create new attendance")
    bigFrameAttNewAtt = Frame(new_attendance, padx=10, pady=10)
    bigFrameAttNewAtt.grid(column=0, row=0)

    Label(bigFrameAttNewAtt, text="Register New Member",
          font=("Arial", 14, "bold")).grid(column=0, row=0)
    formNewAtt = Frame(bigFrameAttNewAtt, bg="grey", padx=10, pady=10)
    formNewAtt.grid(column=0, row=2)
    Label(formNewAtt, text="Name", padx=2).grid(column=0, row=1, padx=5)
    entryInput = Entry(formNewAtt)
    entryInput.grid(column=2, row=1)

    btnSaveNewMember = Button(formNewAtt, text="Save",
                              command=lambda: save_new_member(entryInput, window=new_attendance))
    btnSaveNewMember.grid(column=0, row=2, padx=6, pady=5, sticky="nsew")


def see_member(rootTK):
    new_attendance = Toplevel(rootTK)
    new_attendance.title("Create new attendance")

    bigFrameAttSeeMembers = Frame(new_attendance, bg="white", padx=10, pady=10)
    bigFrameAttSeeMembers.grid(column=0, row=0)
    tableFormSeeMembers = Frame(bigFrameAttSeeMembers, bg="black")
    tableFormSeeMembers.grid(column=0, row=0)
    Label(tableFormSeeMembers, text='No', padx=5, pady=2).grid(
        row=0, column=0, padx=1, pady=1, sticky="nsew")
    Label(tableFormSeeMembers, text='Name', padx=5, pady=2).grid(
        row=0, column=1, padx=1, pady=1, sticky="nsew")

    # global attendances
    for idx in range(len(attendances)):
        Label(tableFormSeeMembers, text=idx+1, padx=5, pady=2).grid(
            row=idx+1, column=0, padx=1, pady=1, sticky="nsew")
        Label(tableFormSeeMembers, text=attendances[idx], padx=5, pady=2).grid(
            row=idx+1, column=1, padx=1, pady=1, sticky="nsew")


def see_attendance(rootTK):
    attendanceScreen = Toplevel(rootTK)
    attendanceScreen.title("Attendance - {}".format(date.today()))

    df = pd.read_csv(
        "kehadiran/attendance_{}.csv".format(date.today()))
    bigFrameAtt = Frame(attendanceScreen, bg="white", padx=10, pady=10)
    bigFrameAtt.grid(column=0, row=0)
    frameAtt = Frame(bigFrameAtt, bg="black")
    frameAtt.grid(row=0, column=0)

    # tampilkan header
    Label(frameAtt, text='Time', padx=5, pady=2).grid(
        row=0, column=1, padx=1, pady=1, sticky="nsew")
    Label(frameAtt, text='Name', padx=5, pady=2).grid(
        row=0, column=2, padx=1, pady=1, sticky="nsew")
    Label(frameAtt, text='Attended', padx=5, pady=2).grid(
        row=0, column=3, padx=1, pady=1, sticky="nsew")

    # tampilkan isi
    for idx, row in df.iterrows():
        Label(frameAtt, text=row['time'], padx=5, pady=2).grid(
            row=idx+1, column=1, padx=1, pady=1, sticky="nsew")
        Label(frameAtt, text=row['name'], padx=5, pady=2).grid(
            row=idx+1, column=2, padx=1, pady=1, sticky="nsew")
        Label(frameAtt, text="Yes" if row['attended'] else "No", padx=5, pady=2).grid(
            row=idx+1, column=3, padx=1, pady=1, sticky="nsew")


def attendance(name):
    today = date.today()
    filename = "kehadiran/attendance_{}.csv".format(today)
    df = pd.read_csv(filename, header=0)

    if not df.loc[df['name'] == name, 'attended'].item():
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        df.loc[df['name'] == name, 'time'] = current_time
        df.loc[df['name'] == name, 'attended'] = True

        df.to_csv(filename, index=False)
        messagebox.showinfo('Attendance Info',
                            "Horay! You're attended for today")


def reset_attendance():
    create_new_attendance_file()
    messagebox.showinfo("Info", "Successfufly reset today attendances")


def generate_image_from_camera(labelLiveCam, labelSnapshot, labelName):
    global choords
    global desc
    # Get the latest frame and convert into Image
    cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    img = cv2.flip(cv2image, 1)
    choords, img = detect(img)
    imgPil = Image.fromarray(img)
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=imgPil.resize((400, 300)))
    labelLiveCam.imgtk = imgtk
    labelLiveCam.configure(image=imgtk)

    if len(choords) > 0:
        imgSnap = crop_image(imgPil, choords, 20)
        imgSnap = imgSnap.resize((256, 256))
        if imgSnap != None:
            imgtkSnap = ImageTk.PhotoImage(image=imgSnap)
            labelSnapshot.imgtk = imgtkSnap
            labelSnapshot.configure(image=imgtkSnap)
            imgGray = imgSnap.convert('L')
            desc = get_featureAKAZE(np.array(imgGray))
            if desc is not None:
                name = who_is_this(desc)
                labelName.config(text=name, font=("Arial", 12, "bold"))
                # labelName.update_idletasks()

                if name != "Tidak dikenali":
                    global start_time
                    if start_time == None:
                        start_time = time.time()
                    if time.time() - start_time > 5:
                        attendance(name)
                        start_time = None

    # Repeat after an interval to capture continiously
    labelLiveCam.after(200, lambda: generate_image_from_camera(
        labelLiveCam, labelSnapshot, labelName))


def create_UI_interface():
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
    submitFormFrame = Frame(infoFrame, highlightbackground="black",
                            highlightthickness=2, padx=2, pady=5)
    submitFormFrame.grid(column=0, row=0, sticky="nsew")

    # create right frame (live webcam and prediction)
    liveCamFrame = Frame(root, width=400, height=600, padx=2, pady=2)
    liveCamFrame.grid(column=2, row=0)

    # create element of middle frame (form)
    Label(submitFormFrame, text="Add new photo", font=(
        "Arial", 16, "bold")).grid(column=0, row=0)
    formFrame = Frame(submitFormFrame, padx=2, pady=2)
    formFrame.grid(column=0, row=1)
    Label(formFrame, text="Name").grid(column=0, row=0)
    stringVar = StringVar()
    defaultValue = "choose a name"
    stringVar.set(defaultValue)
    set_attendance(paramsset_attendance={
                   'formFrame': formFrame,
                   'stringVar': stringVar,
                   'defaultValue': defaultValue,
                   'attendances': attendances})

    btnSave = Button(submitFormFrame, text="Snap and save",
                     command=lambda: save_data(stringVar.get(), desc)
                     )
    btnSave.grid(column=0, row=2, padx=6, sticky="nsew")

    # Create a Label to capture the Video frames in right frame
    Label(liveCamFrame, text="Live WebCam", font=(
        "Arial", 14, "bold")).grid(column=0, row=0, pady=2)
    labelLiveCam = Label(liveCamFrame)
    labelLiveCam.grid(column=0, row=4, sticky=W, pady=2)

    labelName = Label(liveCamFrame, text="person in frame:")
    labelName.grid(column=0, row=2, sticky=W, pady=2)
    labelName = Label(liveCamFrame)
    labelName.grid(column=0, row=3, sticky=W, pady=2)
    labelSnapshot = Label(infoFrame)
    labelSnapshot.grid(column=0, row=4, sticky=W, pady=2)

    # create menu item in left frame
    Label(menuFrame, text="Menu").grid(row=1, column=0, padx=2, pady=2)
    btnAttendance = Button(toolbar, text="See attendance",
                           command=lambda: see_attendance(root)
                           )
    btnAttendance.grid(column=0, row=2, padx=5, pady=5, sticky="nsew")
    btnSeeMembers = Button(toolbar, text="See members",
                           command=lambda: see_member(root)
                           )
    btnSeeMembers.grid(column=0, row=4, padx=5, pady=5, sticky="nsew")
    btnUpdateDataset = Button(
        toolbar, text="Update dataset",
        # command=update_system
    )
    btnUpdateDataset.grid(column=0, row=6, padx=5, pady=5, sticky="nsew")
    btnAddAttendance = Button(toolbar, text="Add attendant",
                              command=lambda: add_new_member(root)
                              )
    btnAddAttendance.grid(column=0, row=8, padx=5, pady=5, sticky="nsew")
    btnReset = Button(toolbar, text="Reset today", bg="firebrick1",
                      command=reset_attendance
                      )
    btnReset.grid(column=0, row=10, pady=5, padx=5, sticky="nsew")

    generate_image_from_camera(labelLiveCam, labelSnapshot, labelName)
    root.mainloop()

    return root
