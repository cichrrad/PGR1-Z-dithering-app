import customtkinter as ctk

#delete/scratch current image
class ImageClose(ctk.CTkButton):
    def __init__(self, parent, close_image_func):
        super().__init__(master = parent, 
                         text='Scrap',
                         command=close_image_func,
                         hover_color='#a81e0c')
        #place
        self.grid(row=0,column=0,sticky='sw',padx=5,pady=45)

#save the current image
class ImageSave(ctk.CTkButton):
    def __init__(self, parent, save_image_func):
        super().__init__(master = parent, 
                         text='Save',
                         hover_color='#098f2c',
                         command=save_image_func)
        #place
        self.grid(row=0,column=0,sticky='sw',padx=5,pady=5)

#menu with effect to be applied to the image        
class EffectMenu(ctk.CTkOptionMenu):
    def __init__(self,parent):
        super().__init__(master = parent,
                         values=["Grayscale","Random dither","Ordered dither","C-dot dither","D-dot dither","Error diffusion","Original"])
        self.grid(row = 0, column=0, sticky = 'nw',pady = 5,padx=5)

#checkbox for enhancing edges
class EdgeEnhance(ctk.CTkCheckBox):
     def __init__(self,parent):
        super().__init__(master = parent,text="Enhance edges")
        self.grid(row = 0, column=0, sticky = 'nw',pady = 45,padx=5)
        
class ApplyEffect(ctk.CTkButton):
    def __init__(self, parent, apply_effect_func):
        super().__init__(master = parent, 
                         text='Apply effect',
                         corner_radius= 5,
                         command=apply_effect_func)
        self.grid(row=0,column=0,sticky='nw',padx=5,pady=85)  

class editMatrices(ctk.CTkButton):
    def __init__(self, parent, edit_matrices_func):
        super().__init__(master = parent, 
                         text='Edit',
                         corner_radius= 5,
                         command=edit_matrices_func)
        self.grid(row=0,column=0,sticky='nw',padx=5,pady=125)

class MatrixWindow(ctk.CTkToplevel):
    def __init__(self,parent, collect_vals_func,currVal):
        super().__init__(master = parent)
        self.title("Edit Matrices used in algorithms")
        self.geometry("300x200")
        self.minsize(200,100)
        self.collect_vals_func = collect_vals_func
        
        self.rowconfigure(0, weight=1)  
        self.columnconfigure(0,weight=1)        
        
        self.ErrorDitherMenu = ctk.CTkOptionMenu(self,values=['Floyd-Steinberg','Stucki','Jarvis, Judice & Ninke','Sierra'])
        self.ErrorDitherMenu.set(currVal)
        self.ErrorDitherLabel = ctk.CTkLabel(self, text="Error diffusion options")
        
        self.ErrorDitherLabel.grid(row = 0, column = 0, sticky = 'n', padx=5,pady=5)
        self.ErrorDitherMenu.grid(row = 0, column = 0,sticky='n',padx=5,pady=45)
        
        self.bind("<Destroy>",collect_vals_func)
