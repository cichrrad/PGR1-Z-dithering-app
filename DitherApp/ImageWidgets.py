import customtkinter as ctk
from tkinter import filedialog, Canvas

#initial popup asking for an image
class ImageLoad(ctk.CTkFrame):
    def __init__(self,parent, load_func):
        super().__init__(master=parent,fg_color='#0c0c0c')
        
        #we want this to fill up the whole window -> we must cover both
        #columns !!! sticky = 'nsew' expands this to full window size
        self.grid(column=0,columnspan=2,row=0,sticky='nsew')
        
        #fetching the load function from the main .py file (not ideal, but it is what it is)
        self.load_func = load_func

        #adding a button that will open file explorer to load an image
        #since we want the button to be in the center, we can simply .pack(expand = True)
        ctk.CTkButton(self,text=' Load image', command=self.load).pack(expand=True)
    
    def load(self):
        
        #using standard open file window to return a path (under name variable)
        path= filedialog.askopenfile()
        if path != None:
            #giving this to the load_func from the main .py file, so that we can use is there
            self.load_func(path.name)
            
        
#canvas showing our image
class ImageCanvas(Canvas):
    def __init__(self,parent,resize_image):
        super().__init__(master = parent, background='#2e2e2e',bd=0,highlightthickness=0,relief='ridge')
        #as shown in the main file, it will be in the 1. column and fill all of it (thus sticky)
        self.grid(row=0,column=1, sticky='nsew')
        #whenever an event Configure (resize) is called, we resize the image
        self.bind('<Configure>',resize_image)
        