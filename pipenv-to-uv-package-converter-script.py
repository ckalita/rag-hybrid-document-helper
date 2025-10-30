import json  # import the JSON module to parse Pipfile.lock contents

with open(
    "Pipfile.lock"
) as f:  # open `Pipfile.lock` for reading using a context manager
    data = json.load(f)  # parse the JSON file into a Python dictionary named `data`

reqs = []  # create an empty list to collect requirement strings

for section in (
    "default",
    "develop",
):  # iterate over the sections to extract packages from
    if section in data:  # check that the section exists in the parsed data
        for pkg, meta in data[
            section
        ].items():  # iterate over package names and their metadata
            version = meta.get(
                "version", ""
            )  # get the `version` field from metadata, default to empty string
            if version.startswith(
                "=="
            ):  # if the version is pinned with `==`, include it in the requirement
                reqs.append(
                    f"{pkg}{version}"
                )  # append the package name with its pinned version to `reqs`
            else:  # if the version is not pinned or missing
                reqs.append(pkg)  # append only the package name to `reqs`

with open(
    "requirements.txt", "w"
) as f:  # open `requirements.txt` for writing using a context manager
    f.write(
        "\n".join(reqs)
    )  # write all collected requirements joined by newlines into the file
