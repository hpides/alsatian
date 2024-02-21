from pkg_resources import working_set


def get_pip_dependency_dict():
    # Create a dictionary to store package names and version numbers
    dependencies = {}

    # Iterate over installed packages and store their names and versions
    for package in working_set:
        package_name = package.key
        package_version = package.version
        dependencies[package_name] = package_version

    return dependencies
