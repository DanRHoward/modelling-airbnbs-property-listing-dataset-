import os
from PIL import Image

def resize_images(): #gathers png images to convert to correct size and store in directory
    file_path = 'C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/images' #path to images to convert
    total_folders_processed = 0 #initial number of image folders that have been converted
    previous_percentage = 0 #initial percentage of folders converted
    total_image_folders = len(os.listdir(file_path)) #number of image folders to convert
    for folder in os.listdir(file_path): #for each image folder in main folder...
        os.chdir(file_path + f'/{folder}') #change working directory to image folder
        total_folders_processed += 1 #add one to variable
        for image in os.listdir(file_path + f'/{folder}'): #for each image png in image folder...
            image_heights = [] #initial list of image heights
            if image.endswith('.png'): #if content of folder ends in '.png' (ie is an image)
                png_image = Image.open(image) #'open' image to be used by python
                png_image_height = png_image.size[1] #collect the height value of the original image
                image_heights.append(png_image_height) #append height of image to out list of heights
        minimum_image_height = min(image_heights) #find minimum height value
        for image in os.listdir(file_path + f'/{folder}'): #for each image png in image folder...
            file_path_to_be_saved = 'C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/processed_images' #file path to save converted images in
            if image.endswith('.png'): #if content of folder ends in '.png' (ie is an image)
                png_image = Image.open(image) #'open' image to be used by python
                resized_width = int(minimum_image_height*png_image.size[0]/png_image.size[1]) #evaluate scaling factor for specific image
                resized_image = png_image.resize((resized_width,minimum_image_height)) #resize image
                os.makedirs(file_path_to_be_saved + f'/{folder}',exist_ok=True) #create the directory for the images to be stored in
                resized_image.save(file_path_to_be_saved + f'/{folder}/{image}.png') #save image in created directory 
            if int(100*total_folders_processed/total_image_folders) % 10 == 0 and int(100*total_folders_processed/total_image_folders) != previous_percentage: #if percentage of folders worked is divisible by 10 and is not equal to the previous iteration...
                print(f'{int(100*total_folders_processed/total_image_folders)}% complete.') #print statement
                previous_percentage = int(100*total_folders_processed/total_image_folders) #set current percentage as previous to be compared
                break #break so as to not continuously print statement (only print it once per 10%)
    return

if __name__ == "__main__": #run code is script
    resize_images()