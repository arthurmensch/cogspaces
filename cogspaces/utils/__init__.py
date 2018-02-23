def zip_data(data, target):
    return {study: (data[study], target[study]) for study in data}


def unzip_data(data):
    return {study: data[study][0] for study in data}, \
           {study: data[study][1] for study in data}