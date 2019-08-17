# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:32:54 2019

@author: JiangZhao
"""

#from Tkinter import *
from tkinter import Tk, Frame, SUNKEN, Scrollbar, HORIZONTAL, E, W, N, S, Canvas, BOTH, ALL
# from tkFileDialog import askopenfilename
from tkinter.filedialog import askopenfilename

from PIL import Image, ImageTk
#from pandas import DataFrame
import numpy as np
import h5py
from keras.models import load_model
model = load_model('LeafRollingScorer_160.h5')
model.summary()

lrs = []#LeafRollingScore
patch_count = 0


if __name__ == "__main__":
    root = Tk()
    root.title("Picture")

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    #adding the image
    File = askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')
    
    
    imgFile0 = Image.open(File)
    scale = 6
    imgData = np.array(imgFile0)
    imgFile = imgFile0.resize([int(imgFile0.size[0]/scale), int(imgFile0.size[1]/scale)], resample = 1)
    
    
    img = ImageTk.PhotoImage(imgFile)
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))


    #function to be called when mouse is clicked
    def func(event):
        #outputting x and y coords to console
        row = event.y
        col = event.x
        # _data: positions for data, not scaled
        row_data = scale * row
        col_data = scale * col
#        print(row_data,col_data)
        
        #get patch
        patchSize = 160
        r_data = int(patchSize/2)
        x1_data = row_data - r_data
        y1_data = col_data - r_data
        patch = imgData[x1_data:x1_data+160, y1_data:y1_data+160, :]
        
        X = np.zeros([1,160,160,3])
        X[0] = patch
        y_temp = model.predict(X).squeeze()*5
        
        global lrs
        global patch_count
        lrs.append(y_temp)
        patch_count += 1
        print('Leaf-Rolling Score ' + str(patch_count) + ': ' + str(y_temp))
        # visulization
        bestColor = "#%02x%02x%02x" % (int(255-np.median(patch[:,:,0])/5), int(255-np.median(patch[:,:,1])), int(255-np.median(patch[:,:,2])))
        r = r_data/scale
        canvas.create_rectangle(col-r, row-r, col+r, row+r, outline='#000000')
#        print(patch.shape)
        canvas.create_text(col-r-5, row-r-5,fill=bestColor,font="Times 10 bold",text=str(patch_count))
        canvas.create_text(col, row,fill=bestColor,font="Times 10 bold",text=str(round(y_temp,2)))
        
#        lrs_df = DataFrame({'Id': range(1, patch_count+1), 'Leaf-rolling score': lrs})
#        lrs_df.to_excel('./LeafRollingScore.xlsx', index=False)
        np.savetxt('../Leaf-rolling scores.csv', lrs, delimiter='\n')

    #mouseclick event
    canvas.bind("<Button 1>",func)

    root.mainloop()
