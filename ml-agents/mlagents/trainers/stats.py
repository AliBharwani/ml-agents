from collections import defaultdict
from enum import Enum
import enum
import pdb
from typing import List, Dict, NamedTuple, Any, Optional
import numpy as np
import abc
import os
import time
from threading import RLock
# from torch.multiprocessing import RLock
import torch.multiprocessing as mp
from mlagents.torch_utils import torch
from mlagents_envs import logging_util
from mlagents_envs.side_channel.stats_side_channel import StatsAggregationMethod

from mlagents_envs.logging_util import get_logger
from mlagents_envs.timers import set_gauge
from torch.utils.tensorboard import SummaryWriter
from mlagents.torch_utils.globals import get_rank

logger = get_logger(__name__)


def _dict_to_str(param_dict: Dict[str, Any], num_tabs: int) -> str:
    """
    Takes a parameter dictionary and converts it to a human-readable string.
    Recurses if there are multiple levels of dict. Used to print out hyperparameters.

    :param param_dict: A Dictionary of key, value parameters.
    :return: A string version of this dictionary.
    """
    if not isinstance(param_dict, dict):
        return str(param_dict)
    else:
        append_newline = "\n" if num_tabs > 0 else ""
        return append_newline + "\n".join(
            [
                "\t"
                + "  " * num_tabs
                + f"{x}:\t{_dict_to_str(param_dict[x], num_tabs + 1)}"
                for x in param_dict
            ]
        )


class StatsSummary(NamedTuple):
    full_dist: List[float]
    aggregation_method: StatsAggregationMethod

    @staticmethod
    def empty() -> "StatsSummary":
        return StatsSummary([], StatsAggregationMethod.AVERAGE)

    @property
    def aggregated_value(self):
        if self.aggregation_method == StatsAggregationMethod.SUM:
            return self.sum
        else:
            return self.mean

    @property
    def mean(self):
        return np.mean(self.full_dist)

    @property
    def std(self):
        return np.std(self.full_dist)

    @property
    def num(self):
        return len(self.full_dist)

    @property
    def sum(self):
        return np.sum(self.full_dist)


class StatsPropertyType(Enum):
    HYPERPARAMETERS = "hyperparameters"
    UNITY_JSON = "unity_json"
    SELF_PLAY = "selfplay"


class StatsWriter(abc.ABC):
    """
    A StatsWriter abstract class. A StatsWriter takes in a category, key, scalar value, and step
    and writes it out by some method.
    """

    def on_add_stat(
        self,
        category: str,
        key: str,
        value: float,
        aggregation: StatsAggregationMethod = StatsAggregationMethod.AVERAGE,
    ) -> None:
        """
        Callback method for handling an individual stat value as reported to the StatsReporter add_stat
        or set_stat methods.

        :param category: Category of the statistics. Usually this is the behavior name.
        :param key: The type of statistic, e.g. Environment/Reward.
        :param value: The value of the statistic.
        :param aggregation: The aggregation method for the statistic, default StatsAggregationMethod.AVERAGE.
        """
        pass

    @abc.abstractmethod
    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        """
        Callback to record training information
        :param category: Category of the statistics. Usually this is the behavior name.
        :param values: Dictionary of statistics.
        :param step: The current training step.
        :return:
        """
        pass

    def add_property(
        self, category: str, property_type: StatsPropertyType, value: Any
    ) -> None:
        """
        Add a generic property to the StatsWriter. This could be e.g. a Dict of hyperparameters,
        a max step count, a trainer type, etc. Note that not all StatsWriters need to be compatible
        with all types of properties. For instance, a TB writer doesn't need a max step.

        :param category: The category that the property belongs to.
        :param property_type: The type of property.
        :param value: The property itself.
        """
        pass


class GaugeWriter(StatsWriter):
    """
    Write all stats that we receive to the timer gauges, so we can track them offline easily
    """

    @staticmethod
    def sanitize_string(s: str) -> str:
        """
        Clean up special characters in the category and value names.
        """
        return s.replace("/", ".").replace(" ", "")

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        for val, stats_summary in values.items():
            set_gauge(
                GaugeWriter.sanitize_string(f"{category}.{val}.mean"),
                float(stats_summary.mean),
            )
            set_gauge(
                GaugeWriter.sanitize_string(f"{category}.{val}.sum"),
                float(stats_summary.sum),
            )


class ConsoleWriter(StatsWriter):
    def __init__(self):
        self.training_start_time = time.time()
        # If self-play, we want to print ELO as well as reward
        self.self_play = False
        self.self_play_team = -1
        self.rank = get_rank()

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        is_training = "Not Training"
        if "Is Training" in values:
            stats_summary = values["Is Training"]
            if stats_summary.aggregated_value > 0.0:
                is_training = "Training"

        elapsed_time = time.time() - self.training_start_time
        log_info: List[str] = [category]
        log_info.append(f"Step: {step}")
        log_info.append(f"Time Elapsed: {elapsed_time:0.3f} s")
        if "Environment/Cumulative Reward" in values:
            stats_summary = values["Environment/Cumulative Reward"]
            if self.rank is not None:
                log_info.append(f"Rank: {self.rank}")

            log_info.append(f"Mean Reward: {stats_summary.mean:0.3f}")
            if "Environment/Group Cumulative Reward" in values:
                group_stats_summary = values["Environment/Group Cumulative Reward"]
                log_info.append(f"Mean Group Reward: {group_stats_summary.mean:0.3f}")
            else:
                log_info.append(f"Std of Reward: {stats_summary.std:0.3f}")
            log_info.append(is_training)

            if self.self_play and "Self-play/ELO" in values:
                elo_stats = values["Self-play/ELO"]
                log_info.append(f"ELO: {elo_stats.mean:0.3f}")
        if "Policy/Loss" in values:
            log_info.append(f"Policy Loss: {values['Policy/Loss'].mean:0.3f}")
        if "World Model/total loss" in values:
            log_info.append(
                f"World Model Loss: {values['World Model/total loss'].mean:0.3f}"
            )
        if "Num Training Updates" in values:
            log_info.append(f"Num Training Updates: {values['Num Training Updates'].full_dist[0]}")
        # else:
        #     log_info.append("No episode was completed since last summary")
        #     log_info.append(is_training)
        for key in values.keys():
            if key.startswith("Supertrack"):
                log_info.append(f"{key}: {values[key].sum}")

        logger.info(". ".join(log_info) + ".")

    def add_property(
        self, category: str, property_type: StatsPropertyType, value: Any
    ) -> None:
        if property_type == StatsPropertyType.HYPERPARAMETERS:
            logger.info(
                """Hyperparameters for behavior name {}: \n{}""".format(
                    category, _dict_to_str(value, 0)
                )
            )
        elif property_type == StatsPropertyType.SELF_PLAY:
            assert isinstance(value, bool)
            self.self_play = value


class TensorboardWriter(StatsWriter):
    def __init__(
        self,
        base_dir: str,
        clear_past_data: bool = False,
        hidden_keys: Optional[List[str]] = None,
    ):
        """
        A StatsWriter that writes to a Tensorboard summary.

        :param base_dir: The directory within which to place all the summaries. Tensorboard files will be written to a
        {base_dir}/{category} directory.
        :param clear_past_data: Whether or not to clean up existing Tensorboard files associated with the base_dir and
        category.
        :param hidden_keys: If provided, Tensorboard Writer won't write statistics identified with these Keys in
        Tensorboard summary.
        """
        self.summary_writers: Dict[str, SummaryWriter] = {}
        self.base_dir: str = base_dir
        self._clear_past_data = clear_past_data
        self.hidden_keys: List[str] = hidden_keys if hidden_keys is not None else []

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        self._maybe_create_summary_writer(category)
        for key, value in values.items():
            if key in self.hidden_keys:
                continue
            self.summary_writers[category].add_scalar(
                f"{key}", value.aggregated_value, step
            )
            if value.aggregation_method == StatsAggregationMethod.HISTOGRAM:
                self.summary_writers[category].add_histogram(
                    f"{key}_hist", np.array(value.full_dist), step
                )
            self.summary_writers[category].flush()
        if "Environment/Interrupted" or "Environment/SelfTerminated" in values:
            # The problem is either of these can be 0, so neither is safe to divide with directly
            num_interrupted = values["Environment/Interrupted"].aggregated_value if "Environment/Interrupted" in values else 0
            num_selfterminated = values["Environment/SelfTerminated"].aggregated_value if "Environment/SelfTerminated" in values else 0
            percent = 0
            if num_selfterminated != 0 or num_interrupted != 0:
                percent = num_interrupted / (num_interrupted + num_selfterminated) * 100
            self.summary_writers[category].add_scalar(
                "Environment/Percent Interrupted", percent, step
            )
            self.summary_writers[category].flush()

    def _maybe_create_summary_writer(self, category: str) -> None:
        if category not in self.summary_writers:
            filewriter_dir = "{basedir}/{category}".format(
                basedir=self.base_dir, category=category
            )
            os.makedirs(filewriter_dir, exist_ok=True)
            if self._clear_past_data:
                self._delete_all_events_files(filewriter_dir)
            self.summary_writers[category] = SummaryWriter(filewriter_dir)

    def _delete_all_events_files(self, directory_name: str) -> None:
        for file_name in os.listdir(directory_name):
            if file_name.startswith("events.out"):
                logger.warning(
                    f"Deleting TensorBoard data {file_name} that was left over from a "
                    "previous run."
                )
                full_fname = os.path.join(directory_name, file_name)
                try:
                    os.remove(full_fname)
                except OSError:
                    logger.error(
                        "{} was left over from a previous run and "
                        "not deleted.".format(full_fname)
                    )

    def add_property(
        self, category: str, property_type: StatsPropertyType, value: Any
    ) -> None:
        if property_type == StatsPropertyType.HYPERPARAMETERS:
            assert isinstance(value, dict)
            summary = _dict_to_str(value, 0)
            self._maybe_create_summary_writer(category)
            if summary is not None:
                self.summary_writers[category].add_text("Hyperparameters", summary)
                self.summary_writers[category].flush()


class StatsReporterABC(abc.ABC):

    @abc.abstractmethod
    def add_property(self, property_type: StatsPropertyType, value: Any) -> None:
        pass

    @abc.abstractmethod
    def add_stat(
        self,
        key: str,
        value: float,
        aggregation: StatsAggregationMethod = StatsAggregationMethod.AVERAGE,
    ) -> None:
        pass

    @abc.abstractmethod
    def set_stat(self, key: str, value: float) -> None:
        pass

    @abc.abstractmethod
    def write_stats(self, step: int) -> None:
        pass

class StatsReporter(StatsReporterABC):
    writers: List[StatsWriter] = []
    stats_dict: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    lock = RLock()
    stats_aggregation: Dict[str, Dict[str, StatsAggregationMethod]] = defaultdict(
        lambda: defaultdict(lambda: StatsAggregationMethod.AVERAGE)
    )

    def __init__(self, category: str):
        """
        Generic StatsReporter. A category is the broadest type of storage (would
        correspond the run name and trainer name, e.g. 3DBalltest_3DBall. A key is the
        type of stat it is (e.g. Environment/Reward). Finally the Value is the float value
        attached to this stat.
        """
        self.category: str = category

    @staticmethod
    def add_writer(writer: StatsWriter) -> None:
        # with StatsReporter.lock:
        StatsReporter.writers.append(writer)

    def add_property(self, property_type: StatsPropertyType, value: Any) -> None:
        """
        Add a generic property to the StatsReporter. This could be e.g. a Dict of hyperparameters,
        a max step count, a trainer type, etc. Note that not all StatsWriters need to be compatible
        with all types of properties. For instance, a TB writer doesn't need a max step.

        :param property_type: The type of property.
        :param value: The property itself.
        """
        with StatsReporter.lock:
            for writer in StatsReporter.writers:
                writer.add_property(self.category, property_type, value)

    def add_stat(
        self,
        key: str,
        value: float,
        aggregation: StatsAggregationMethod = StatsAggregationMethod.AVERAGE,
    ) -> None:
        """
        Add a float value stat to the StatsReporter.

        :param key: The type of statistic, e.g. Environment/Reward.
        :param value: the value of the statistic.
        :param aggregation: the aggregation method for the statistic, default StatsAggregationMethod.AVERAGE.
        """
        with StatsReporter.lock:
            StatsReporter.stats_dict[self.category][key].append(value)
            StatsReporter.stats_aggregation[self.category][key] = aggregation
            for writer in StatsReporter.writers:
                writer.on_add_stat(self.category, key, value, aggregation)

    def set_stat(self, key: str, value: float) -> None:
        """
        Sets a stat value to a float. This is for values that we don't want to average, and just
        want the latest.

        :param key: The type of statistic, e.g. Environment/Reward.
        :param value: the value of the statistic.
        """
        with StatsReporter.lock:
            StatsReporter.stats_dict[self.category][key] = [value]
            StatsReporter.stats_aggregation[self.category][
                key
            ] = StatsAggregationMethod.MOST_RECENT
            for writer in StatsReporter.writers:
                writer.on_add_stat(
                    self.category, key, value, StatsAggregationMethod.MOST_RECENT
                )

    def write_stats(self, step: int) -> None:
        """
        Write out all stored statistics that fall under the category specified.
        The currently stored values will be averaged, written out as a single value,
        and the buffer cleared.

        :param step: Training step which to write these stats as.
        """
        with StatsReporter.lock:
            values: Dict[str, StatsSummary] = {}
            for key in StatsReporter.stats_dict[self.category]:
                if len(StatsReporter.stats_dict[self.category][key]) > 0:
                    stat_summary = self._get_stats_summaries(key)
                    values[key] = stat_summary
            for writer in StatsReporter.writers:
                writer.write_stats(self.category, values, step)
            del StatsReporter.stats_dict[self.category]

    def _get_stats_summaries(self, key: str) -> StatsSummary:
        """
        Get the mean, std, count, sum and aggregation method of a particular statistic, since last write.

        :param key: The type of statistic, e.g. Environment/Reward.
        :returns: A StatsSummary containing summary statistics.
        """
        stat_values = StatsReporter.stats_dict[self.category][key]
        if len(stat_values) == 0:
            return StatsSummary.empty()

        return StatsSummary(
            full_dist=stat_values,
            aggregation_method=StatsReporter.stats_aggregation[self.category][key],
        )

class StatsReporterCommand(enum.Enum):
    ADD_STAT = 1
    SET_STAT = 2
    WRITE_STATS = 3
    ADD_PROPERTY = 4

def stats_processor(category : str, queue: mp.Queue, writers: List[StatsWriter]):
    stats_dict: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    stats_aggregation: Dict[str, Dict[str, StatsAggregationMethod]] = defaultdict(
        lambda: defaultdict(lambda: StatsAggregationMethod.AVERAGE)
    )        
    # Set log level. On some platforms, the logger isn't common with the
    # main process, so we need to set it again.
    logging_util.set_log_level(logging_util.INFO)

    try:
        while True:
            _queried = False
            while not queue.empty():
                _queried = True
                command, args = queue.get()
                if command == StatsReporterCommand.ADD_STAT:
                    key, value, aggregation = args
                    stats_dict[category][key].append(value)
                    stats_aggregation[category][key] = aggregation
                elif command == StatsReporterCommand.SET_STAT:
                    key, value = args
                    stats_dict[category][key] = [value]
                    stats_aggregation[category][key] = StatsAggregationMethod.MOST_RECENT
                elif command == StatsReporterCommand.WRITE_STATS:
                    step = args
                    values: Dict[str, StatsSummary] = {}
                    for key in stats_dict[category]:
                        if len(stats_dict[category][key]) > 0:
                            stat_summary = StatsSummary(
                                full_dist=stats_dict[category][key],
                                aggregation_method=stats_aggregation[category][key],
                            )
                            values[key] = stat_summary
                    for writer in writers:
                        writer.write_stats(category, values, step)
                    del stats_dict[category]
                elif command == StatsReporterCommand.ADD_PROPERTY:
                    property_type, value = args
                    for writer in writers:
                        writer.add_property(category, property_type, value)
            if not _queried:
                # Yield thread to avoid busy-waiting
                time.sleep(.001)
    except(KeyboardInterrupt) as ex:
        logger.debug("StatsReporter shutting down.")
    except Exception as ex:
        logger.exception("An unexpected error occurred in the StatsReporter.")
    finally:
        queue.close()
        queue.join_thread()
        logger.info("StatsReporter closing.")



class StatsReporterMP(StatsReporterABC):
    
    def __init__(self, category: str, queue: mp.Queue):
        """
        Generic StatsReporter. A category is the broadest type of storage (would
        correspond the run name and trainer name, e.g. 3DBalltest_3DBall. A key is the
        type of stat it is (e.g. Environment/Reward). Finally the Value is the float value
        attached to this stat.
        """
        self.category: str = category
        self.queue = queue

    def add_stat(
        self,
        key: str,
        value: float,
        aggregation: StatsAggregationMethod = StatsAggregationMethod.AVERAGE,
    ) -> None:
        """
        Add a float value stat to the StatsReporter.

        :param key: The type of statistic, e.g. Environment/Reward.
        :param value: the value of the statistic.
        :param aggregation: the aggregation method for the statistic, default StatsAggregationMethod.AVERAGE.
        """
        try:
            self.queue.put((StatsReporterCommand.ADD_STAT, (key, value, aggregation)))
        except Exception as ex:
            logger.exception(f"An unexpected error occurred in the StatsReporter: {ex}")

    def set_stat(self, key: str, value: float) -> None:
        try:
            self.queue.put((StatsReporterCommand.SET_STAT, (key, value)))
        except Exception as ex:
            logger.exception(f"An unexpected error occurred in the StatsReporter: {ex}")

    def write_stats(self, step: int) -> None:
        try:
            self.queue.put((StatsReporterCommand.WRITE_STATS, (step)))
        except Exception as ex:
            logger.exception(f"An unexpected error occurred in the StatsReporter: {ex}")

    def add_property(self, property_type: StatsPropertyType, value: Any) -> None:
        try:
            self.queue.put((StatsReporterCommand.ADD_PROPERTY, (property_type, value)))
        except Exception as ex:
            logger.exception(f"An unexpected error occurred in the StatsReporter: {ex}")
