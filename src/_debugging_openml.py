"""
Something suddenly changed in either the OpenML API or the website.
"""
import openml
from openml.study.study import BaseStudy, OpenMLBenchmarkSuite, OpenMLStudy

import xmltodict

# this is a desperation move, cutting and pasting the code to see if we can figure out what is wrong
def _get_study(id_: int | str, entity_type: str) -> BaseStudy:
    xml_string = openml._api_calls._perform_api_call(f"study/{id_}", "get")
    force_list_tags = (
        "oml:data_id",
        "oml:flow_id",
        "oml:task_id",
        "oml:setup_id",
        "oml:run_id",
        "oml:tag",  # legacy.
    )
    result_dict = xmltodict.parse(xml_string, force_list=force_list_tags)["oml:study"]
    study_id = int(result_dict["oml:id"])
    alias = result_dict.get("oml:alias", None)
    main_entity_type = result_dict["oml:main_entity_type"]

    if entity_type != main_entity_type:
        raise ValueError(
            f"Unexpected entity type '{main_entity_type}' reported by the server"
            f", expected '{entity_type}'"
        )

    benchmark_suite = result_dict.get("oml:benchmark_suite", None)
    name = result_dict["oml:name"]
    description = result_dict["oml:description"]
    status = result_dict["oml:status"]
    creation_date = result_dict["oml:creation_date"]
    creator = result_dict["oml:creator"]

    # tags is legacy. remove once no longer needed.
    tags = []
    if "oml:tag" in result_dict:
        for tag in result_dict["oml:tag"]:
            current_tag = {"name": tag["oml:name"], "write_access": tag["oml:write_access"]}
            if "oml:window_start" in tag:
                current_tag["window_start"] = tag["oml:window_start"]
            tags.append(current_tag)

    def get_nested_ids_from_result_dict(key: str, subkey: str) -> list[int] | None:
        """Extracts a list of nested IDs from a result dictionary.

        Parameters
        ----------
        key : str
            Nested OpenML IDs.
        subkey : str
            The subkey contains the nested OpenML IDs.

        Returns
        -------
        Optional[List]
            A list of nested OpenML IDs, or None if the key is not present in the dictionary.
        """
        if result_dict.get(key) is not None:
            return [int(oml_id) for oml_id in result_dict[key][subkey]]
        return None

    datasets = get_nested_ids_from_result_dict("oml:data", "oml:data_id")
    tasks = get_nested_ids_from_result_dict("oml:tasks", "oml:task_id")

    if main_entity_type in ["runs", "run"]:
        flows = get_nested_ids_from_result_dict("oml:flows", "oml:flow_id")
        setups = get_nested_ids_from_result_dict("oml:setups", "oml:setup_id")
        runs = get_nested_ids_from_result_dict("oml:runs", "oml:run_id")

        study = OpenMLStudy(
            study_id=study_id,
            alias=alias,
            benchmark_suite=benchmark_suite,
            name=name,
            description=description,
            status=status,
            creation_date=creation_date,
            creator=creator,
            tags=tags,
            data=datasets,
            tasks=tasks,
            flows=flows,
            setups=setups,
            runs=runs,
        )  # type: BaseStudy

    elif main_entity_type in ["tasks", "task"]:
        study = OpenMLBenchmarkSuite(
            suite_id=study_id,
            alias=alias,
            name=name,
            description=description,
            status=status,
            creation_date=creation_date,
            creator=creator,
            tags=tags,
            data=datasets,
            tasks=tasks,
        )

    else:
        raise ValueError(f"Unknown entity type {main_entity_type}")

    return study

if __name__ == "__main__":
    suite = _get_study(353, "tasks")