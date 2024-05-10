#GUI imports===================================================================
import customtkinter as ctk
#links to tkinter docs
#https://customtkinter.tomschimansky.com/documentation/
#https://customtkinter.tomschimansky.com/tutorial/
#https://docs.python.org/3/library/tkinter.ttk.html#module-tkinter.ttk
#==============================================================================
#also depends on 'packaging'


#Files=========================================================================
from ImageWidgets import *
from ControlWidgets import *
from matrixDefinitions import *
#==============================================================================

#Other Imports=================================================================
from PIL import Image, ImageFilter, ImageTk
from scipy.ndimage import convolve
import numpy as np

import random
import time
#==============================================================================


#wrapper function used to time functions
def timing_wrapper(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"{func.__name__} took {elapsed_time:.2f} milliseconds to run.")
            return result
        return wrapper

#Main class of our project - contains the window, global variables, event functions,...
class root(ctk.CTk):
    
    #CONSTRUCTOR========================================================================================================
    def __init__(self):
        

        #===============================================================================================================
        #config of the main window======================================================================================
        #===============================================================================================================
        super().__init__()
        
        #default mode to launch the app in (we dont do light mode here)
        ctk.set_appearance_mode('dark')
        
        #setting initial size and boundary sizes for the window
        self.geometry('1000x600')
        self.minsize(800,600)
        self.title('DitherDemo')
        
        #===============================================================================================================
        #===============================================================================================================
        #===============================================================================================================
        


        #===============================================================================================================
        #layout=========================================================================================================
        #===============================================================================================================
        
        #The main layout of our GUI: [1 row x 2 column] grid:
        #+-------------+--------------------------------------+
        #|             |                                      |
        #|             |                                      |
        #|             |                                      |
        #|             |                                      |
        #|             |                                      |
        #|             |                                      |
        #| [Controls]  |           [Picture view]             |
        #|             |                                      |
        #|             |                                      |
        #|             |                                      |
        #|             |                                      |
        #|             |                                      |
        #|             |                                      |
        #+-------------+--------------------------------------+
        #we must weight the grid accordingly
        self.rowconfigure(0, weight=1) 
        
        self.columnconfigure(0,minsize=100,weight=0) #weight of 0 so that this column does not resize (in X axis) when we stretch the window
        self.columnconfigure(1, weight= 1)    
        
        #===============================================================================================================
        #===============================================================================================================
        #===============================================================================================================
        


        #===============================================================================================================
        #global variables===============================================================================================
        #===============================================================================================================
        
        #variables to track window's dimension to apply effects correctly
        #without these, the effect will be show only once we move/resize the window after applying it
        self.imageW=0
        self.imageH=0
        self.canvasW=0
        self.canvasH=0
        
        #variables related to the currently loaded imaged --> set during load_image func
        self.imageOG = None
        self.image = None
        self.imagetk = None
        self.imageRatio = None

        #used for the pop-up window to configure matrices, when starting the program, there is no instance of this window, thus None
        self.editMatrixWindow = None        

        #Default value of the ordered dither matrix
        self.orderedDitherMatrix = np.array([[0,32,8,40,2,34,10,42],
                                             [48,16,56,24,50,18,58,26],
                                             [12,44,4,36,14,46,6,38],
                                             [60,28,52,20,62,30,54,22],
                                             [3,35,11,43,1,33,9,41],
                                             [51,19,59,27,49,17,57,25],
                                             [15,47,7,39,13,45,5,37],
                                             [63,31,55,23,61,29,53,21]],dtype=np.uint8)
        
        #Default value of the c-dot ordered dither matrix
        self.clusterDotDitherMatrix = np.array([[0,2],
                                               [3,1]],dtype=np.uint8)
        

        #Default value of the d-dot ordered dither matrix
        self.dispersedDotDitherMatrix = np.array([[0,12,3,15],
                                                [8,4,11,7],
                                                [2,14,1,13],
                                                [10,6,9,5]],dtype=np.uint8)
        
        #Default value of the error distribution matrix
        self.errorDistributionMatrix = ERROR_MATRICES["Stucki"]
        
        #Default value for edge enhancing - unused for faster implementations
        self.enhanceEdgesFactor = 0.45;
        
        #===============================================================================================================
        #===============================================================================================================
        #===============================================================================================================
        


        #===============================================================================================================
        #widgets========================================================================================================
        #===============================================================================================================
        
        #Initial 'Load Image' button prompt
        self.imageLoadPrompt = ImageLoad(self,self.load_image)
        
        #===============================================================================================================
        #===============================================================================================================
        #===============================================================================================================
      
        #run
        self.mainloop()
    #===================================================================================================================
        



    #FUNCTIONS========================================================================================================== 
    
    #loads the image and shows the UI
    def load_image(self,path):
        
        #once we have the path from the ImageLoad widget, use it to load an image
        #keep original image in imageOG
        self.imageOG = Image.open(path)
        self.image = self.imageOG
        
        #getting ratio for proper scaling of the image ([0]-width,[1]-height)
        self.imageRatio = self.image.size[0] / self.image.size[1]
        
        #variable used to draw the image
        self.imagetk= ImageTk.PhotoImage(self.image)
        
        #when an image is loaded, we should hide the widget asking to load an image (duh)
        self.imageLoadPrompt.grid_forget()
        
        #create an ImageCanvas to show the image (placing handled by resize_image)
        self.imageCanvas = ImageCanvas(self,self.resize_image)
        
        #show the Controls
        self.effectMenu = EffectMenu(self)
        self.edgeEnhanceButton = EdgeEnhance(self)
        self.applyEffectButton = ApplyEffect(self, self.apply_effect)
        self.unloadButton = ImageClose(self, self.close_image)
        self.saveButton = ImageSave(self,self.save_image)
        self.editButton = editMatrices(self, self.edit_matrices)
    
    #saves the current edit of the image
    def save_image(self):
        imageTobeSaved=ImageTk.getimage(self.imagetk)
        #save dialog
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        
        # Check if the user canceled the dialog
        if not file_path:
            return
        imageTobeSaved.save(file_path)
    
    #resizes the image to have correct ratios
    def resize_image(self,event):
        
        #get current ratio of the canvas and decide how to resize the image
        cRatio = event.width / event.height
        
        self.canvasW = event.width
        self.canvasH = event.height

        if cRatio > self.imageRatio: #width of the canvas > width of the image
            #keep image height and resize accordingly
            self.imageH =int(event.height)
            self.imageW =int(self.imageH*self.imageRatio)
        else:                        #height of the canvas > image
            #keep image width and resize accordingly
            self.imageW=int(event.width)
            self.imageH =int(self.imageW/self.imageRatio)
        self.draw_image()
    
    #deletes old image, draws a new one using global canvas variables
    def draw_image(self):
        
        self.imageCanvas.delete('all')
        resizedImage = self.image.resize((self.imageW,self.imageH))
        self.imagetk = ImageTk.PhotoImage(resizedImage)
        #place the new image, with correct size and position 
        self.imageCanvas.create_image( self.canvasW /2 , self.canvasH / 2, image = self.imagetk)
    
    #unloads ('scraps') current image and shows load prompt again
    def close_image(self):
        #hide the image
        self.imageCanvas.grid_forget()
        #hide the controls
        self.unloadButton.grid_forget()
        self.saveButton.grid_forget()
        self.effectMenu.grid_forget()
        self.edgeEnhanceButton.grid_forget()
        self.editButton.grid_forget()
        #show the load prompt again
        self.imageLoadPrompt = ImageLoad(self,self.load_image)
    
    #opens the matrix edit pop-up (if there is none)
    def edit_matrices(self):
         if self.editMatrixWindow is None or not self.editMatrixWindow.winfo_exists():
            self.editMatrixWindow = MatrixWindow(self,collect_vals_func=self.collect_vals,currVal=self.errorDistributionMatrix) # create window if its None or destroyed
         else:
            return
        
    #collects values from the matrix edit pop-up (bound to <destroy> event - calls itself multiple times, idk why, but it does not really matter)
    def collect_vals(self,event):
        vals = self.editMatrixWindow.ErrorDitherMenu.get()
        self.errorDistributionMatrix = vals
        
    #EFFECTS============================================================================================================
    #alternative solutions are commented out - many are more intuitive and make way more sense when reading the code
    #BUT they are MUCH MUCH slower most of the time
    

    #switch-like function to determine what effect to apply
    def apply_effect(self):
        vals=["Grayscale","Random dither","Ordered dither","C-dot dither","D-dot dither","Error diffusion","Original"]
        
        if (self.effectMenu.get() == vals[0]):
            self.image = self.imageOG
            self.effect_grayscale()
            self.draw_image()
        elif(self.effectMenu.get() == vals[1]):
            self.image = self.imageOG
            self.effect_randomDither()
            self.draw_image()
        elif(self.effectMenu.get() == vals[2]):
            self.image = self.imageOG
            self.effect_orderedDither()
            self.draw_image()
        elif(self.effectMenu.get() == vals[3]):
            self.image = self.imageOG
            self.effect_cDotDither()
            self.draw_image()
        elif(self.effectMenu.get() == vals[4]):
            self.image = self.imageOG
            self.effect_dDotDither()
            self.draw_image()      
        elif(self.effectMenu.get() == vals[5]):
            self.image = self.imageOG
            self.effect_errorDifusion()
            self.draw_image()
        elif(self.effectMenu.get() == vals[-1]):
            self.image = self.imageOG
            self.draw_image()
    
    #function to treshold image based on a given value x - not used, as this can be a one-liner in python
    def treshold(self,x):
        
        self.effect_grayscale()
        imageArr = np.array(self.image)
        imageArr = 255 * (imageArr>x)
        self.image = Image.fromarray(imageArr)  

    #enhances edges
    def enhance_edges(self):
        
        #deselect the enhance button and apply grayscale
        self.edgeEnhanceButton.deselect()
        self.effect_grayscale()
        
        #fetch the image
        imageArr = np.array(self.image)
        
        # [1] using custom convolve kernel =================================================================================================================
    
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])

        # Make sure the image and kernel have the same data type
        imageArr = imageArr.astype(np.float32)
        # Apply convolution
        result = convolve(imageArr, kernel)

        # Clip values to be in the valid range [0, 255]
        result = np.clip(result, 0, 255)
        imageArr = result
        

        #[2] using convolve kernel - fastest, but cannot be simply modified via constant=====================================================================
        #https://en.wikipedia.org/wiki/Kernel_(image_processing)

        # imageArr = imageArr.filter(ImageFilter.Kernel((3, 3), [
        # -1, -1, -1,
        # -1, 9, -1,
        # -1, -1, -1,
        # ], scale=1))

        
        #[3] "naive" implementation - too slow for 'only' enhancing edges, but can be adjusted via constant + is kinda weird================================

        # imageArr = np.array(self.image, dtype=np.uint16)
        # imageArr = np.pad(imageArr,pad_width=1,mode='constant',constant_values=0)
        # height, width = imageArr.shape
        # alpha = self.enhanceEdgesFactor

        # edgeEnhancedArr = np.zeros_like(imageArr,dtype=np.uint16)
        # for y in range(1,height-1):
        #     for x in range(1, width-1):
        #         edgeEnhancedArr[y][x]= imageArr[y][x]+imageArr[y-1][x-1]+imageArr[y-1][x]+imageArr[y-1][x+1]+imageArr[y][x+1]+imageArr[y+1][x+1]+imageArr[y+1][x]+imageArr[y+1][x-1]+imageArr[y][x-1]
        #         edgeEnhancedArr[y][x]= edgeEnhancedArr[y][x]/9
        #         edgeEnhancedArr[y][x]= (imageArr[y][x] -(alpha * edgeEnhancedArr[y][x]))/(1-alpha)
   
        # edgeEnhancedArr = edgeEnhancedArr[1:-1,1:-1]
        
        #set the image to our new image
        
        imageArr = imageArr.astype("uint8")   
        self.image=Image.fromarray(imageArr)
    
    @timing_wrapper    
    def effect_grayscale(self):
        
       if self.edgeEnhanceButton.get() == 1 : 
            self.enhance_edges()
       else:
            
            gImage = self.image
            gImage = gImage.convert("L")
             
            #[1] pixel-by-pixel approach - way too slow=====================================================================================================
        
            # gImage = Image.new("L", self.image.size)
            # for x in range(gImage.width):
            #     for y in range(gImage.height):
            #         #luminance formula
            #         gImage.putpixel((x,y),int(0.2989 * self.image.getpixel((x,y))[0] + 0.587 * self.image.getpixel((x,y))[1] + 0.114 * self.image.getpixel((x,y))[2]))
        
            #[2] numpy approach - faster, but still slow====================================================================================================        

            # imageArr = np.array(self.image)
            # gImage = np.zeros((self.image.height, self.image.width))
            # height, width = gImage.shape

            # for y in range(height):
            #     for x in range(width):
            #         gImage[y][x] = int(0.2989 * imageArr[y][x][0] + 0.587 * imageArr[y][x][1] + 0.114 * imageArr[y][x][2])
              
            # self.image = Image.fromarray(gImage.astype(np.uint8))

            self.image=gImage

        
    @timing_wrapper
    def effect_randomDither(self):
        
        if self.edgeEnhanceButton.get() == 1 : 
            self.enhance_edges()    
        else:
            self.effect_grayscale()
        
        #fetch the image as array (more like matrix)
        imageArr = np.array(self.image)
        #populate same-sized matrix with random numbers 0-1
        randArr = np.random.rand(*imageArr.shape)
        #compare with normalized image pixel intensity (True = 1, False = 0, duh)
        image = (randArr < imageArr / 255).astype("uint8")

        self.image = Image.fromarray(image*255)

        #numpy free method - why are you doing this to yourself?
        
        # image = Image.new("L",self.image.size)
        # for x in range(image.width):
        #     for y in range(image.height):
        #         pixel = self.image.getpixel((x,y))
        #         rand = random.uniform(0,255)
        #         if (rand < pixel):
        #             image.putpixel((x,y),255)
        #         else:
        #             image.putpixel((x,y),0)                  
        # self.image=image
    
    @timing_wrapper
    def effect_orderedDither(self):
         
         if self.edgeEnhanceButton.get() == 1 : 
            self.enhance_edges()
         else:
            self.effect_grayscale()
         
        
         #Pixel-by-Pixel method - not terribly slow, but slow nonetheless   
         
         # orderSize = self.orderDitherThresholdMap.shape[0]
        
         # imageArr = np.array(self.image,dtype=np.uint8)
         # height , width = imageArr.shape
        
         # for y in range(height):
         #    for x in range(width):
         #        i,j = y % orderSize, x % orderSize
         #        res =  ((self.orderDitherThresholdMap[i][j] + 0.5)/(orderSize ** 2))*255
         #        imageArr[y][x] = 255 * (imageArr[y][x] >= res)


         #Leveraging numPy operations and meshgrid - way faster (~420 times) than Pixel-by-Pixel method

         orderMapSize = self.orderedDitherMatrix.shape[0]

         imageArr = np.array(self.image, dtype=np.uint8)
         thresholdMap = ((self.orderedDitherMatrix + 0.5) / (orderMapSize ** 2)) * 255

         # Create coordinate matrices
         y, x = np.meshgrid(np.arange(imageArr.shape[0]), np.arange(imageArr.shape[1]), indexing='ij')

         # Perform thresholding
         imageArr = 255 * (imageArr >= thresholdMap[y % orderMapSize, x % orderMapSize])
         imageArr = imageArr.astype("uint8")
         self.image = Image.fromarray(imageArr)


    @timing_wrapper
    def effect_cDotDither(self):
          
         if self.edgeEnhanceButton.get() == 1 : 
            self.enhance_edges()
         else:
            self.effect_grayscale()
         
         # [0] Pixel-by-Pixel method - slow 
         
         # imageArr = np.array(self.image, dtype = np.uint8)
         # height, width = imageArr.shape

         # # Normalize the clusterMatrix
         # clusteredMatrixSize = self.clusteredDotDitherMap.shape[0]
         # clusterMatrix = (self.clusteredDotDitherMap / clusteredMatrixSize**2 ) * 255

         # for y in range(height):
         #    for x in range(width):
         #        # Calculate the threshold based on the dither matrix
         #        threshold = clusterMatrix[y % (clusteredMatrixSize) , x % (clusteredMatrixSize)]

         #        # Apply dithering
         #        imageArr[y, x] = (imageArr[y, x] > threshold)* 255

         # self.image = Image.fromarray(imageArr)
         
         #[1]Leveraging numPy operations and meshgrid - way faster (~420 times) than Pixel-by-Pixel method

         orderMapSize = self.clusterDotDitherMatrix.shape[0]

         imageArr = np.array(self.image, dtype=np.uint8)
         
         #normalize the threshold matrix to be 0-1, then multiply by 255 to be 0-255, +0.5 for rounding (also changes brigthness)
         thresholdMap = ((self.clusterDotDitherMatrix+0.5) / (orderMapSize ** 2)) * 255

         #create coordinates - without them, we would have to go pixel-per-pixel
         #with them we can perform element-wise operations and leverage numpy vectorization 
         y, x = np.meshgrid(np.arange(imageArr.shape[0]), np.arange(imageArr.shape[1]), indexing='ij')

         # Perform thresholding (fast thanks to coordinates)
         imageArr = 255 * (imageArr >= thresholdMap[y % orderMapSize, x % orderMapSize])
         imageArr = imageArr.astype("uint8")
         self.image = Image.fromarray(imageArr)
    
    
    @timing_wrapper
    def effect_dDotDither(self):
        
          
         if self.edgeEnhanceButton.get() == 1 : 
            self.enhance_edges()
         else:
            self.effect_grayscale()
                   
         # imageArr = np.array(self.image, dtype = np.uint8)
         # height, width = imageArr.shape

         # # Normalize the clusterMatrix
         # clusteredMatrixSize = self.clusteredDotDitherMap.shape[0]
         # clusterMatrix = (self.clusteredDotDitherMap / clusteredMatrixSize**2 ) * 255

         # for y in range(height):
         #    for x in range(width):
         #        # Calculate the threshold based on the dither matrix
         #        threshold = clusterMatrix[y % (clusteredMatrixSize) , x % (clusteredMatrixSize)]

         #        # Apply dithering
         #        imageArr[y, x] = (imageArr[y, x] > threshold)* 255

         # self.image = Image.fromarray(imageArr)
         
         #Leveraging numPy operations and meshgrid - way faster (~420 times) than Pixel-by-Pixel method

         orderMapSize = self.dispersedDotDitherMatrix.shape[0]

         imageArr = np.array(self.image, dtype=np.uint8)
         
         #normalize the threshold matrix to be 0-1, then multiply by 255 to be 0-255, +0.5 for rounding (also changes brigthness)
         thresholdMap = ((self.dispersedDotDitherMatrix + 0.5) / (orderMapSize ** 2)) * 255

         #create coordinates - without them, we would have to go pixel-per-pixel
         #with them we can perform element-wise operations and leverage numpy vectorization
         y, x = np.meshgrid(np.arange(imageArr.shape[0]), np.arange(imageArr.shape[1]), indexing='ij')

         # Perform thresholding
         imageArr = 255 * (imageArr >= thresholdMap[y % orderMapSize, x % orderMapSize])
         imageArr = imageArr.astype("uint8")
         self.image = Image.fromarray(imageArr)
    
    @timing_wrapper
    def effect_errorDifusion(self):
       
        if self.edgeEnhanceButton.get() == 1 : 
            self.enhance_edges()
        else:
            self.effect_grayscale()
        
        image_arr = np.array(self.image, dtype=np.int16)
        height, width = image_arr.shape
        
        
        if self.errorDistributionMatrix == "Floyd-Steinberg":
            
            #pad the image to not have to boundary check every pixel
            image_arr = np.pad(image_arr, 1, mode='constant', constant_values=0)
            
            for y in range(1,height+1):
                for x in range(1,width+1):
                    old_pixel = image_arr[y, x]
                    new_pixel = 255 * (old_pixel >= 127)
                    #pre-divide = better to divide one time that to multiply by fraction 4 times
                    quantization_error = (old_pixel - new_pixel)/16
                    image_arr[y, x] = new_pixel

                    # Diffusion to neighboring pixels
                    image_arr[y, x + 1] += quantization_error * 7
                    image_arr[y + 1, x - 1] += quantization_error * 3 
                    image_arr[y + 1, x] += quantization_error * 5
                    image_arr[y + 1, x + 1] += quantization_error * 1
            
            #remove padding
            image_arr = image_arr[1:-1, 1:-1]
                    
        elif self.errorDistributionMatrix == "Stucki":
            
            image_arr = np.pad(image_arr, 2, mode='constant', constant_values=0)
            
            for y in range(2,height+2):
                for x in range(2,width+2):
                    old_pixel = image_arr[y, x]
                    new_pixel = 255 * (old_pixel >= 127)
                    quantization_error = (old_pixel - new_pixel)/42
                    image_arr[y, x] = new_pixel

                    # Diffusion to neighboring pixels
                    image_arr[y, x + 1] += quantization_error * 8 
                    image_arr[y , x + 2] += quantization_error * 4 
                    
                    image_arr[y + 1, x-2] += quantization_error * 2 
                    image_arr[y + 1, x-1] += quantization_error * 4 
                    image_arr[y + 1, x] += quantization_error * 8
                    image_arr[y + 1, x+1] += quantization_error * 4
                    image_arr[y + 1, x+2] += quantization_error * 2

                    image_arr[y + 2, x-2] += quantization_error * 1 
                    image_arr[y + 2, x-1] += quantization_error * 2 
                    image_arr[y + 2, x] += quantization_error * 4
                    image_arr[y + 2, x+1] += quantization_error * 2
                    image_arr[y + 2, x+2] += quantization_error * 1

            image_arr = image_arr[2:-2, 2:-2]
                    
        elif self.errorDistributionMatrix == "Jarvis, Judice & Ninke":
            
            image_arr = np.pad(image_arr, 2, mode='constant', constant_values=0)
            
            for y in range(2,height + 2):
                for x in range(2,width + 2):
                    old_pixel = image_arr[y, x]
                    new_pixel = 255 * (old_pixel >= 127)
                    quantization_error = (old_pixel - new_pixel)/48
                    image_arr[y, x] = new_pixel

                    # Diffusion to neighboring pixels
                    image_arr[y, x + 1] += quantization_error * 7 
                    image_arr[y , x + 2] += quantization_error * 5 
                    
                    image_arr[y + 1, x-2] += quantization_error * 3
                    image_arr[y + 1, x-1] += quantization_error * 5 
                    image_arr[y + 1, x] += quantization_error * 7
                    image_arr[y + 1, x+1] += quantization_error * 5
                    image_arr[y + 1, x+2] += quantization_error * 3

                    image_arr[y + 2, x-2] += quantization_error * 1 
                    image_arr[y + 2, x-1] += quantization_error * 3
                    image_arr[y + 2, x] += quantization_error   * 5
                    image_arr[y + 2, x+1] += quantization_error * 3
                    image_arr[y + 2, x+2] += quantization_error * 1
                    
            image_arr = image_arr[2:-2, 2:-2]
                    
        elif self.errorDistributionMatrix == "Sierra":
 
            image_arr = np.pad(image_arr, 1, mode='constant', constant_values=0)
            
            for y in range(1,height + 1):
                for x in range(1,width + 1):
                    old_pixel = image_arr[y, x]
                    new_pixel = 255 * (old_pixel >= 127)
                    quantization_error = (old_pixel - new_pixel)
                    image_arr[y, x] = new_pixel

                    # Diffusion to neighboring pixels
                    image_arr[y, x + 1] += quantization_error * 0.5 
                    image_arr[y + 1, x - 1] += quantization_error * 0.25 
                    image_arr[y + 1, x] += quantization_error * 0.25 
        
            image_arr = image_arr[1:-1, 1:-1]

        image_arr = np.clip(image_arr, 0, 255)
        image_arr = image_arr.astype("uint8")        
        self.image = Image.fromarray(image_arr)
                  

#call our root
root()  

