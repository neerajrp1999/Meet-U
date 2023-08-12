from tkinter import *

def close_window():
  global running
  running = False  # turn off while loop
  print( "Window closed")

root = Tk()
root.protocol("WM_DELETE_WINDOW", close_window)
cv = Canvas(root, width=200, height=200)
cv.pack()

running = True
# This is an endless loop stopped only by setting 'running' to 'False'
while running: 
  for i in range(200): 
    if not running: 
        break
    cv.create_oval(i, i, i+1, i+1)
    root.update() 