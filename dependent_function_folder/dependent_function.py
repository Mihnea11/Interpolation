import pathlib
import numpy
import dependent_function_folder.generate_data as generate_data
import dependent_function_folder.data_prediction as data_prediction
import dependent_function_folder.create_data_files as create_data_files


def function_dependence_interpolation():
    default_folder = "data"
    current_path = pathlib.Path(__file__).parent.resolve().joinpath(default_folder)

    raw_data = generate_data.generate_raw_data(current_path)
    raw_copy = raw_data.copy()

    create_data_files.create_input_file(current_path, raw_copy)
    raw_copy = raw_data.copy()
    create_data_files.create_target_file(current_path, raw_copy)

    predicted = data_prediction.predict_dependence_data(current_path, raw_data)
    success_rate = numpy.corrcoef(predicted[0], predicted[5])

    return success_rate
