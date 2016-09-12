import os

def load( labels_dir, object_name ):
    # Positions of the bouding box pixels inside each label file
    col_first = 4
    row_first = 5
    col_last = 6
    row_last = 7
    label_files = os.listdir( labels_dir )
    label_dataset = []
    print 'Loading ' + object_name + ' labels... '
    for label_index, label_name in enumerate( label_files ):
        label_file = os.path.join( labels_dir, label_name)
        # print 'Loading label file:', label_file # -> todo save this info to file
        with open( label_file ) as fileHandle:
            file_content = fileHandle.readlines()
            # file_content is a list of strings and may have several lines
            for i in range(len(file_content)):
                obstacle = file_content[i].split()
                if ( obstacle[0] == object_name ):
                    # Create a tuple with file name and bounding box pixels
                    bounding_box = [ label_name, [obstacle[row_first], obstacle[row_last], obstacle[col_first], obstacle[col_last]] ]
                    # Save tuple to array of tuples
                    label_dataset.append( bounding_box )
    return label_dataset