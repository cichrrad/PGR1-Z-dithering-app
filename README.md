wq---

# DitherApp Readme

## CONTENTS

### [1] General Information
### [2] User Interface
### [3] Dependencies

## [1] GENERAL INFORMATION

The DitherApp consists of 4 .py files:

- **DitherApp.py**: The main file (**TO LAUNCH THE APPLICATION, RUN THIS FILE**) containing the `root` class - the application is an instance of this class. It also includes functions performing basic operations (load/save) and algorithms for image modification.

- **ImageWidgets.py**: Contains GUI elements related to the initial image loading and the GUI element on which the image is displayed (Canvas).

- **ControlWidgets.py**: Contains GUI elements related to controlling the program (left control column with buttons). The logic/functions adding functionality to these elements are in DitherApp.py.

- **matrixDefinitions.py**: Contains various matrices for Error diffusion.

The program allows the user to load an image (in standard formats) and apply one of the following effects:
- Grayscale
- Random dither
- Ordered dither
- C-dot dither
- D-dot dither
- Error diffusion
- Original

Each effect (except Original) allows the option to 'enhance edges'. For all effects except Grayscale and Original, Grayscale is applied first. The user can change the settings for the 'Error diffusion' effect. The user can also save the modified image, discard it without saving, and load a new image.

## [2] USER INTERFACE

The UI of the application has 3 main parts:

- **Initial 'Load Image' screen**: The first thing the user sees upon startup - a prompt to load an image.

- **Control panel**: A panel with buttons for controlling the program. The user can select an effect from a dropdown menu, check whether to enhance edges, and click 'Apply effect' to apply the effect. The user can also click the edit button, which opens a pop-up window with a dropdown menu. Here the user can set how the error will be distributed in the error diffusion effect.

- **Display panel**: Panel displaying the image with the applied effect.

## [3] DEPENDENCIES

The application requires Python version 3.7 or higher.
Dependencies generated by the 'pip freeze' command:

customtkinter version 5.2.1
darkdetect version 0.8.0
networkx version 3.2.1 
numpy version 1.26.1
packaging version 23.2
Pillow version 10.1.0
scipy version 1.11.3

Primary uses of modules:
- **CustomTkinter ([link](https://customtkinter.tomschimansky.com/))**: An extension of the standard GUI module tkinter. Used for a nicer GUI and Dark theme. Installation: `pip install customtkinter`.

- **PIL ([link](https://pillow.readthedocs.io/en/stable/))**: Python Image Library. Used for image manipulation in ImageTk format (for Tkinter elements). *Was not used to apply effects. Installation: `pip install Pillow`.

- **SciPy ([link](https://scipy.org/))**: Open-source software for mathematics, science, and engineering. Along with NumPy, it is used to accelerate and optimize algorithms. Installation: `pip install scipy`.

- **NumPy ([link](https://numpy.org/))**: See SciPy. Installation: `pip install numpy`.

The application also utilizes these (hopefully) standard modules:

- random
- time - for time wrapper measuring runtime functions
- packaging
- tkinter

---
