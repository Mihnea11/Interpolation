import pathlib
import GenerateData
import DataPrediction
import CreateDataFiles


def main():
    defaultFolder = "Data"
    currentPath = pathlib.Path(__file__).parent.resolve().joinpath(defaultFolder)

    rawData = GenerateData.generate_data(currentPath)
    rawCopy = rawData.copy()

    CreateDataFiles.create_input_file(currentPath, rawCopy)
    rawCopy = rawData.copy()
    CreateDataFiles.create_target_file(currentPath, rawCopy)

    DataPrediction.predict_data(currentPath, rawData)

if __name__ == '__main__':
    main()
