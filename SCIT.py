print "Begin Imports"
print "Importing os"
import os

print "Importing ar"
import arrow as ar

print "Importing np"
import numpy as np

print "Importing JSON"
import json as jn

print "Importing zlib"
import zlib as zlib

print "Importing png"
import png

print "Importing fabs"
from math import fabs

print "Importing measure"
from skimage import measure

print "Importing matplotlib"
from matplotlib import pyplot as plt

print "Importing ndimage"
from scipy import ndimage

print "Done"


class ObservationPeriod:

    def __init__(self, path):
        self.base_path = path
        self.frame_dictionary = {}
        self.list_file_codes = []

        self.identify_file_codes()

        for i in self.list_file_codes:
            fn = self.base_path + "\\" + i[:-4] + '.obj'
            try:
                with open(fn, 'rb') as f:
                    print
                    print "Decoding", i
                    self.decode_frame_analysis(f, i)
                    print "Done"
            except IOError:
                print
                print "Analysing", i
                self.create_frame_analysis()
                print "Done"

    def identify_file_codes(self):

        for i in os.listdir(self.base_path):
            if os.path.basename(i)[-4:] == ".png" and os.path.basename(i)[:3] == "IDR":
                self.list_file_codes.append(os.path.basename(i))

    def create_frame_analysis(self):
        for i in self.list_file_codes:
            self.frame_dictionary[i] = FrameAnalysis(self.base_path + "\\" + i, 512, 6, False)

    def analyse_tracks(self):
        pass

    def encode_frame_analysis(self):
        print "Begin Serialization"
        for i in self.list_file_codes:
            fn = self.base_path + "\\" + i[:-4] + '.obj'

            encode_dictionary = {}
            encode_dictionary['frame_path'] = self.frame_dictionary[i].frame_path
            encode_dictionary['fn'] = self.frame_dictionary[i].fn
            encode_dictionary['time_stamp_begin'] = self.frame_dictionary[i].time_stamp_begin
            encode_dictionary['time_stamp_end'] = self.frame_dictionary[i].time_stamp_end
            encode_dictionary['file_code'] = self.frame_dictionary[i].file_code
            encode_dictionary['time_start'] = self.frame_dictionary[i].time_start
            encode_dictionary['pixels_wh'] = self.frame_dictionary[i].pixels_wh
            encode_dictionary['total_volume'] = self.frame_dictionary[i].total_volume
            encode_dictionary['average_direction'] = self.frame_dictionary[i].average_direction
            encode_dictionary['average_distance'] = self.frame_dictionary[i].average_distance
            encode_dictionary['list_cells'] = self.frame_dictionary[i].list_cells
            encode_dictionary['list_cell_steps'] = self.frame_dictionary[i].encode_cell_steps()

            np.savez_compressed(fn[:-4], matrix_intensity=self.frame_dictionary[i].matrix_intensity, matrix_cells=self.frame_dictionary[i].matrix_cells, matrix_binary=self.frame_dictionary[i].matrix_binary)

            print "Saving"
            with open(fn, 'wb') as f:
                json_data = jn.dumps(encode_dictionary)
                compressed_data = zlib.compress(json_data)
                f.write(compressed_data)

    def decode_frame_analysis(self, f, i):
        print "Opening File"
        compressed_data = f.read()
        print "Decompressing Data"
        json_data = zlib.decompress(compressed_data)
        print "Creating Dictionary"
        decoded_dictionary = jn.loads(json_data)

        print "Assigning Variables"
        frame_path = self.base_path + "\\" + i
        pixels_wh = decoded_dictionary['pixels_wh']
        frame_length_mins = (int(decoded_dictionary['time_stamp_end']) - int(decoded_dictionary['time_stamp_begin']))/60
        array_frame = np.load(frame_path[:-4] + '.npz')
        
        self.frame_dictionary[i] = FrameAnalysis(frame_path, pixels_wh, frame_length_mins, True)
        
        self.frame_dictionary[i].frame_path = decoded_dictionary['frame_path']
        self.frame_dictionary[i].fn = decoded_dictionary['fn']
        self.frame_dictionary[i].time_stamp_begin = decoded_dictionary['time_stamp_begin']
        self.frame_dictionary[i].time_stamp_end = decoded_dictionary['time_stamp_end']
        self.frame_dictionary[i].file_code = decoded_dictionary['file_code']
        self.frame_dictionary[i].time_start = decoded_dictionary['time_start']
        self.frame_dictionary[i].pixels_wh = decoded_dictionary['pixels_wh']
        self.frame_dictionary[i].total_volume = decoded_dictionary['total_volume']
        self.frame_dictionary[i].average_direction = decoded_dictionary['average_direction']
        self.frame_dictionary[i].average_distance = decoded_dictionary['average_distance']
        self.frame_dictionary[i].matrix_intensity = array_frame['matrix_intensity']
        self.frame_dictionary[i].matrix_cells = array_frame['matrix_cells']
        self.frame_dictionary[i].matrix_binary = array_frame['matrix_binary']
        self.frame_dictionary[i].list_cells = decoded_dictionary['list_cells']

        array_cells_intensity = np.load(frame_path[:-4] + '-cells-intensity.npz')
        array_cells_binary = np.load(frame_path[:-4] + '-cells-binary.npz')

        k = 0
        print "Creating Cell Steps"
        # print "Cells:", len(array_cells['array_matrix_binary'])

        arr_list = []
        for l in array_cells_intensity.items():
            arr_list.append(int(l[0][4:]))
            arr_list.sort()

        for j in arr_list:
            k += 1
            cell_steps_dictionary = decoded_dictionary['list_cell_steps'][j]

            cell_id = cell_steps_dictionary['cell_id']
            cell_number = cell_steps_dictionary['cell_number']

            temp_cell_step = CellStep(array_cells_intensity['arr_' + str(j)], cell_id, cell_number, True)

            temp_cell_step.pre_cells = cell_steps_dictionary['pre_cells']
            temp_cell_step.post_cells = cell_steps_dictionary['post_cells']
            temp_cell_step.volume = cell_steps_dictionary['volume']
            temp_cell_step.matrix_binary = array_cells_binary['arr_' + str(j)]
            temp_cell_step.centroid = cell_steps_dictionary['centroid']

            self.frame_dictionary[i].list_cell_steps.append(temp_cell_step)


class FrameAnalysis:

    def __init__(self, frame_path, pixels_width, frame_length_mins, defrost):

        self.frame_path = frame_path
        self.fn = os.path.basename(frame_path)[:-4]
        self.time_stamp_begin = ar.get(self.fn.split('.')[2], 'YYYYMMDDHHmm').timestamp
        self.time_stamp_end = self.time_stamp_begin + frame_length_mins * 60
        self.file_code = os.path.basename(frame_path).split('.')[2]
        self.time_start = ar.get(os.path.basename(frame_path).split('.')[2], 'YYYYMMDDHHmm').timestamp
        self.pixels_wh = pixels_width

        self.total_volume = 0
        self.average_direction = 0
        self.average_distance = 0
        self.matrix_intensity = ""
        self.matrix_cells = ""
        self.matrix_binary = ""
        self.list_cells = []
        self.list_cell_steps = []

        if defrost:
            pass
        else:
            self.load_frame_into_matrix()
            # 60 because 128km radar
            self.identify_cells_from_matrix(4, 30)
            self.create_cells_from_matrix()
        
    def load_frame_into_matrix(self):

        # A simple function that finds the mean
        def find_mean(value_list):
            k = 0.0
            for i_val in value_list:
                k += i_val
            return k / len(value_list)

        # PNG File has a key map as to the colour, with each pixel assigned a number corresponding to the colour
        # First need to create a map of these colours and what radar intensity they represent.
        palette_key_temp = {'(245, 245, 255, 255)': 1,
                            '(180, 180, 255, 255)': 2,
                            '(120, 120, 255, 255)': 3,
                            '(20, 20, 255, 255)': 4,
                            '(0, 216, 195, 255)': 5,
                            '(0, 150, 144, 255)': 6,
                            '(0, 102, 102, 255)': 7,
                            '(255, 255, 0, 255)': 8,
                            '(255, 200, 0, 255)': 9,
                            '(255, 150, 0, 255)': 10,
                            '(255, 100, 0, 255)': 11,
                            '(255, 0, 0, 255)': 12,
                            '(200, 0, 0, 255)': 13,
                            '(120, 0, 0, 255)': 14,
                            '(40, 0, 0, 255)': 15}

        # Need to create a map that will correspond the intensity to what ever the PNG index number is
        palette_key_reversed = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
                                15: 0}

        # This map is group together, basically use an average of the whole map to ensure that there are not two
        # indexes that are unintentionally assigned. Thisi s what i_avg is for

        i_avg = []

        # Open the file with the PNG reader:
        r = png.Reader(self.frame_path)
        read = r.read()

        # Read through the palette key from the PNG - read[3]['palette']
        for i in range(0, len(read[3]['palette'])):

            # Check through the palette key representing the intensity colours
            for j in palette_key_temp:

                # If there is a match (ie. the PNG palette key represents an intensity colour
                if str(read[3]['palette'][i]) == str(j):

                    # Take note of the index in the PNG.
                    # This will be used to ensure all indices are close to each other (in a bunch) remducing and
                    # almost removing the possibility of doubling up on hte PNG indices reoccurring
                    i_avg.append(i)

        # Find the mean of the indices
        i_avg = find_mean(i_avg)

        # Go through the PNG palette again
        for i in range(0, len(read[3]['palette'])):

            # Go through the locally defined palette
            for j in palette_key_temp:

                # If there is a match between the local palette and the PNG palette
                if str(read[3]['palette'][i]) == str(j):

                    # Make sure that the PNG and local palette are not duplicates
                    if fabs(i - i_avg) < 16:

                        # Add the key/value combination to be used to decode the PNG
                        palette_key_reversed[palette_key_temp[j]] = i

        # Flip the palette key around to give the PNG palette to return the intensity value
        palette_key = dict((v, k) for k, v in palette_key_reversed.iteritems())

        # Load the data into a matrix
        matrix_radar_initial_load = np.vstack(read[2])

        # Copy the initial matrix into the working matrix
        matrix_radar_working = np.copy(matrix_radar_initial_load)

        # Work through the working matrix, substitute the PNG palette for intensity values
        for k, v in palette_key.iteritems():
            matrix_radar_working[matrix_radar_working == k] = v

        # Zero everything else
        temp_matrix = np.asarray(matrix_radar_working)
        high_values_indices = temp_matrix > 15  # Where values are low
        low_values_indices = temp_matrix < 0  # Where values are low
        temp_matrix[high_values_indices] = 0  # All low values set to 0
        temp_matrix[low_values_indices] = 0  # All low values set to 0

        # Copy matrix into final
        # self.matrix_intensity = np.flipud(temp_matrix)
        self.matrix_intensity = temp_matrix
        self.total_volume = self.matrix_intensity.sum()

    def identify_cells_from_matrix(self, threshold, pixel_threshold):
        # Need to figure out what size clusters to throw out
        # Suggestion is 30 for 256kms, 60 for 128kms and 90 for 64kms

        # Threshold is not for initial identification of cells, this is done by listing intensity above 4
        # Threshold is for what intensity would like to be kept in the matrix

        # This is the threshold for identifying storms, i.e. Radar intensity above this will be I.D.ed as a storm cell
        radar_matrix_binary = np.copy(self.matrix_intensity)

        # For elements above threshold, make value of 1
        radar_matrix_binary[self.matrix_intensity > threshold] = 1
        # All other elements equal zero
        radar_matrix_binary[self.matrix_intensity <= threshold] = 0
        
        self.matrix_binary = radar_matrix_binary

        # Now there is a binary matrix of cells and not cells.
        # Need to identify the blobs, this is done by the measure function.
        matrix_radar_working = measure.label(radar_matrix_binary)

        # Very complicated bit of code to figure out whether a cluster is actually not rain
        cluster_list = matrix_radar_working.ptp()
        for i in range(0, cluster_list + 1):
            if len(self.matrix_intensity[matrix_radar_working == i]) > pixel_threshold:
                ave_intensity = float((self.matrix_intensity[matrix_radar_working == i]).sum()) /\
                      len(self.matrix_intensity[matrix_radar_working == i])
                if ave_intensity < 5:
                    matrix_radar_working[matrix_radar_working == i] = 0

        # This code removes these 'patches' to create homogeneous blobs.
        radar_matrix_binary = np.copy(matrix_radar_working)
        radar_matrix_binary[radar_matrix_binary > 0] = 1
        radar_matrix_binary[radar_matrix_binary < 1] = 0
        matrix_radar_working = measure.label(radar_matrix_binary)

        # Very complicated bit of code to figure out whether a cluster is actually not rain
        cluster_list = matrix_radar_working.ptp()
        for i in range(0, cluster_list + 1):
            if len(self.matrix_intensity[matrix_radar_working == i]) > pixel_threshold:
                ave_intensity = float((self.matrix_intensity[matrix_radar_working == i]).sum()) / \
                                len(self.matrix_intensity[matrix_radar_working == i])
                if ave_intensity < 5:
                    matrix_radar_working[matrix_radar_working == i] = 0

        # Identify the number of clusters
        max_number_of_clusters_found = matrix_radar_working.ptp()

        # Cycle through each cluster
        for i in range(0, max_number_of_clusters_found + 1):

            # Print cluster ID and number of pixels.
            # print i, (self.matrix_radar_working == i).sum()

            # If the number of pixels in the cluster is smaller than threshold
            if (matrix_radar_working == i).sum() < pixel_threshold:

                # Keep everything but that cluster
                radar_matrix_first = np.copy(matrix_radar_working)
                radar_matrix_binary[radar_matrix_first == i] = 0
                radar_matrix_binary[radar_matrix_binary > 0] = 1

        matrix_radar_working = measure.label(radar_matrix_binary)

        # Very complicated bit of code to figure out whether a cluster is actually not rain
        cluster_list = matrix_radar_working.ptp()
        for i in range(0, cluster_list + 1):
            if len(self.matrix_intensity[matrix_radar_working == i]) > pixel_threshold:
                ave_intensity = float((self.matrix_intensity[matrix_radar_working == i]).sum()) / \
                                len(self.matrix_intensity[matrix_radar_working == i])
                if ave_intensity < 5:
                    matrix_radar_working[matrix_radar_working == i] = 0

        # Set the Cells Matrix in self
        self.matrix_cells = np.copy(matrix_radar_working)

        # Need to create a list of the cell numbers
        list_raw_cells = np.unique(self.matrix_cells)
        list_cells = []

        # Need to cycle through all cells and remove anything that is not rain (radar intensity low)
        for i in list_raw_cells:
            if len(self.matrix_intensity[self.matrix_cells == i]) > pixel_threshold:
                ave_intensity = float((self.matrix_intensity[self.matrix_cells == i]).sum()) / \
                                len(self.matrix_intensity[self.matrix_cells == i])
                if ave_intensity >= 5:
                    list_cells.append(int(i))

        self.list_cells = list_cells

    def create_cells_from_matrix(self):

        for i in self.list_cells:

            tempmatrix = np.copy(self.matrix_intensity)
            cellid = i
            tempmatrix[self.matrix_cells == cellid] = 1
            tempmatrix[self.matrix_cells != cellid] = 0

            working_matrix = np.copy(self.matrix_intensity)
            working_matrix[tempmatrix == 0] = 0

            cell_id = str(self.file_code) + str(i)

            self.list_cell_steps.append(CellStep(working_matrix, cell_id, i, False))

    def encode_cell_steps(self):

        encode_list = []
        array_matrix_intensity = []
        array_matrix_binary = []
        for i in self.list_cell_steps:

            encode_dictionary = {}
            encode_dictionary['pre_cells'] = i.pre_cells
            encode_dictionary['post_cells'] = i.post_cells
            encode_dictionary['volume'] = i.volume
            array_matrix_intensity.append(i.matrix_intensity)
            array_matrix_binary.append(i.matrix_binary)
            encode_dictionary['cell_id'] = i.cell_id
            encode_dictionary['cell_number'] = i.cell_number
            encode_dictionary['centroid'] = i.centroid
            encode_list.append(encode_dictionary)

        fn_intensity = self.frame_path[:-4] + '-cells-intensity'
        fn_binary = self.frame_path[:-4] + '-cells-binary'

        array_len = len(array_matrix_intensity)
        np.savez_compressed(fn_intensity, *array_matrix_intensity[:array_len])
        np.savez_compressed(fn_binary, *array_matrix_binary[:array_len])

        return encode_list

    def other_stuff(self):

        # plt.imshow(self.matrix_binary, cmap='spectral')
        # plt.imshow(self.matrix_intensity, cmap='jet')
        plt.imshow(self.matrix_cells, cmap='spectral')
        plt.show()


class CellStep:

    def __init__(self, matrix_intensity, cell_id, cell_number, defrost):
        self.pre_cells = []
        self.post_cells = []

        # Below Values are worked out on creation of object
        self.volume = 0
        self.matrix_intensity = matrix_intensity
        self.matrix_binary = ""

        self.cell_id = cell_id
        self.cell_number = cell_number
        self.centroid = ""

        if defrost:
            pass
        else:
            self.create_binary_matrix()
            self.find_centroid()
            self.get_volume()

    def create_binary_matrix(self):

        # Need to load the intensity matrix first

        working_matrix = np.copy(self.matrix_intensity)
        self.matrix_binary = np.copy(self.matrix_intensity)

        # Find intensities greater than 4, make 1
        # Everything else make 0
        self.matrix_binary[working_matrix > 4] = 1
        self.matrix_binary[working_matrix <= 4] = 0

    def find_centroid(self):

        # Load Matrix to work with
        working_matrix = self.matrix_binary

        # Load working matrix for editing
        temp_matrix = np.copy(working_matrix)

        # Calculate the centroid
        yx = ndimage.measurements.center_of_mass(temp_matrix)

        self.centroid = (yx[1], yx[0])

    def get_volume(self):
        self.volume = self.matrix_intensity.sum()


class Terminal:

    def __init__(self, term_id):

        self.cell_centroid = 0
        self.term_centroid = 0
        self.term_x = 0
        self.term_y = 0
        self.term_ID = term_id


if __name__ == "__main__":

    # frame_test = FrameAnalysis("C:\Users\Nathan\Documents\Storm Chasing\\temp2\IDR023.T.201611210601.png", 512, 6)

    # plt.imshow(frame_test.matrix_intensity, cmap='jet')
    # plt.show()

    obs1 = ObservationPeriod("C:\Users\Nathan\Documents\Storm Chasing\\temp2\\")
    # obs1.encode_frame_analysis()

