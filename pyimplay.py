import ctypes
import numpy as np
import cv2
import colorsys

def get_screen_size() -> int:
    """Windows: Finds the main display size through windows API

    Returns:
        int, int: display pixel sizes in x and y
    """
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

def implay(imgs: list, window_title: str="implay", loop: bool=False, fps: int=10, show_index: bool=True):
    """Simple image player based on OpenCV. Takes a stack of images 

    Controls:
        l: Change between manual and looping mode

        Manual mode:
            a,d: Change image index (a => -1, d => +1)
            1,2,3,4,5: Jump between fifths of the image stack

        Looping mode:
            w,s: Change the fps (w => +1, s => -1)

    Args:
        imgs (list): List of images, where the images numpy array. Should be normalized to 0-255
        window_title (str, optional): Title of the window. Defaults to "implay".
        loop (bool, optional): Select the initial mode. Defaults to False.
        fps (int, optional): Select the initial frames per second. Defaults to 10.
        show_index (bool, optional): Select whether to display the current image index on top left corner. Defaults to True.
    """
    # Initialize image stack
    imgs = np.array(imgs).astype(np.uint8)
    h_img,w_img = imgs.shape[1:3]

    # Print image stack info
    print("\n== Image player ==")
    print("Image stack parameters:")
    print("min: ", np.min(imgs[0]))
    print("max: ", np.max(imgs[0]))
    print("imgs shape: ", imgs.shape)
    print("dtype: ", imgs.dtype)
    print()
    
    # Get display size (on Windows) and initialize window at 75% of display size height
    w_screen,h_screen = get_screen_size()
    img_aspect_ratio = w_img/h_img
    w_screen = h_screen*img_aspect_ratio
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title,int(w_screen*0.75),int(h_screen*0.75))

    # Initialize text parameters
    org = (w_img//100,h_img//20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = w_img//1000
    color = (255, 0, 0)
    thickness = w_img//1000

    # Initialize looping parameters
    loop_ascending = True
    i = 0
    key = ""
    while key != ord("q"):
        # Print info
        if loop:
            print(f"Looping mode | frame i: {i} | press w/s to change looping speed ({fps} fps)    ", end="\r")
        else:
            print(f"Manual mode | Press a/d to move between images, viewing image i={i}", end="\r")
        
        # Show image
        disp_image = imgs[i] #cv2.putText(imgs[i], f"{i}", (3,15), font, fontScale, color, thickness)
        if show_index:
            disp_image = cv2.putText(disp_image, f"i: {i}", org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow(window_title,disp_image)
        
        # Change i in looping mode
        if loop:
            key = cv2.waitKey(round(1000/fps))
            if i == len(imgs)-1:
                loop_ascending = False
            elif i == 0:
                loop_ascending = True

            if loop_ascending:
                i += 1
            else:
                i -= 1

            if key == ord("w"):
                fps = fps + 1 if fps < 200 else 200
            if key == ord("s"):
                fps = fps - 1 if fps > 1 else 1
            
        # Change i in manual mode
        else:
            key = cv2.waitKey(0)
            if key == ord("d"):
                i += 1 if i < len(imgs)-1 else 0
            elif key == ord("a"):
                i -= 1 if i > 0 else 0

            # Skip to different portion of the image stack by pressing 0,1,2,3,4,5
            if len(imgs) > 5:
                if key == ord("1"):
                    i = 0
                elif key == ord("2"):
                    i = int(len(imgs) * (1/4))
                elif key == ord("3"):
                    i = int(len(imgs) * (2/4))
                elif key == ord("4"):
                    i = int(len(imgs) * (3/4))
                elif key == ord("5"):
                    i = len(imgs) - 1
                

        if key == ord("l"):
            loop = abs(loop-1)
            print("\x1b[K", end="\r")
    
    cv2.destroyWindow(window_title)

if __name__=="__main__":
    """ Usage example

        Generates an image stack with numpy and gives the image list to implay

    """
    # Generate sample image stack
    imglist = []
    res = 512
    n = 200
    diameter = 20
    offset_rad = 50
    for i in range(n):
        value = (np.array(colorsys.hsv_to_rgb(i/n,1,1))*255).astype(np.uint8)
        img = np.zeros((res,res,3))
        x,y = np.sin((i/n)*2*np.pi)*offset_rad, np.cos((i/n)*2*np.pi)*offset_rad
        y,x = round(y+res//2), round(res//2+x)
        h1,h2 = y-diameter, y+diameter
        w1,w2 = x-diameter, x+diameter
        img[h1:h2,w1:w2] = value
        imglist.append(img)

    # Give the images to implay
    implay(imglist, fps=100, loop=1)
