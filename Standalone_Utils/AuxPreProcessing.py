from abc import ABC, abstractmethod

import numpy as np


class ReadAddresses(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    # for testing purposes only
    def lbas_read(self, slba, nlb, time_stamp_us):
        pass

    @abstractmethod
    def lbas_write(self, slba, nlb, time_stamp_us):
        pass

    @abstractmethod
    # num of lbas in the data structure
    def get_lbas(self):
        pass

    @abstractmethod
    def get_size(self):
        pass

    def balance_size(self):
        print('Note: "balance_size()" is not implemented for this class!')


class PRG(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def rand_range(self, rng):
        pass


# noinspection PyPep8Naming
class PRG_LFSR(PRG):
    def __init__(self, seed):
        self.state = seed

    def generate_k_bits(self, k):
        bits = ''
        for ind in range(k):
            bit = (self.state ^ (self.state >> 1) ^ (self.state >> 3) ^ (self.state >> 12)) & 1
            bits += str(bit)
            self.state = (self.state >> 1) | (bit << 15)
        return bits

    def rand_range(self, rng):
        return int(self.generate_k_bits(10), 2) % rng

    def n_choose_2(self, rng):
        if rng == 3:
            return [
                [0, 1],
                [0, 2],
                [1, 0],
                [1, 2],
                [2, 0],
                [2, 1]
            ][self.rand_range(6)]
        print(f'rand_perm is not supported for range = {rng}')

class CuckooHashTable:
    def __init__(self, params):
        self.params = params
        self.buckets: list = [[None, None]] * int(self.params["hash_size"])  # [key, value]
        self.prg = PRG_LFSR(5)
        self.num_of_items = 0
        self.max_num_of_items = 0  # for research purposes.
        self.a_0 = 888600117
        self.b_0 = 1218133043
        self.a_1 = 2025020841
        self.b_1 = 748081337
        self.a_2 = 553470329
        self.b_2 = 688321842
        self.big_prime = 2257957291
        self.num_of_hashes = 3
        self.max_search_length = 100
        self.h = [self.h_0(), self.h_1(), self.h_2()]

    def h_0(self):
        return lambda key: ((key * self.a_0 + self.b_0) % self.big_prime) % int(self.params["hash_size"])

    def h_1(self):
        return lambda key: ((key * self.a_1 + self.b_1) % self.big_prime) % int(self.params["hash_size"])

    def h_2(self):
        return lambda key: ((key * self.a_2 + self.b_2) % self.big_prime) % int(self.params["hash_size"])

    def __del__(self):
        # print(f'hash deleted. max num of items was {self.max_num_of_items}')
        pass

    def __contains__(self, key):
        hash_num = 0
        while hash_num < self.num_of_hashes:
            # go over all slot functions
            slot = self.h[hash_num](key)
            if self.buckets[slot][0] == key:
                return True
            hash_num += 1
        return False

    def __setitem__(self, key, value):
        # print(f'Cuckoo load factor: {self.num_of_items / ReadAddressesSegmentsChunkedDict.hash_size}')
        hash_num = 0
        available_bucket_num = False
        while hash_num < self.num_of_hashes:
            # go over all hash functions
            bucket_num = self.h[hash_num](key)
            if self.buckets[bucket_num][0] == key:  # overwrite
                self.buckets[bucket_num] = [key, value]
                return True, 0
            if self.buckets[bucket_num][0] is None and not available_bucket_num:  # first available slot
                available_bucket_num = bucket_num + 1  # +1 so that 0 will be true...
            hash_num += 1
        if available_bucket_num:
            self.buckets[available_bucket_num - 1] = [key, value]
            self.num_of_items += 1
            return True, 0
        else:
            #  Choose kicked out pair.
            rand_hash = self.prg.rand_range(self.num_of_hashes)
            slot_ind = self.h[rand_hash](key)
            [kicked_key, kicked_value] = self.buckets[slot_ind]
            self.buckets[slot_ind] = [key, value]
            return self._insert(kicked_key, kicked_value, slot_ind, 1)

    def _insert(self, key, value, evacuated_slot, path_len):
        hash_num = 0
        while hash_num < self.num_of_hashes:
            # go over all hash functions; one of them will point to the slot we came from, but there's no problem with that - it will be occupied.
            if self.buckets[self.h[hash_num](key)][0] is None:
                self.buckets[self.h[hash_num](key)] = [key, value]
                break
            hash_num += 1
        if hash_num < self.num_of_hashes:  # success
            self.num_of_items += 1
            return True, path_len

        # evacuated item too has no place. Choose next victim to kick out.
        if path_len == self.max_search_length:
            # print('^^^ failed!')
            # theoretically, can return value of kicked out chunk so its segments can be removed from FIFO, but actually there is no problem with them staying there.
            return False, None
        #  Choose next slot.

        # old implementation
        while True:
            rand_hash = self.prg.rand_range(self.num_of_hashes)
            new_slot_ind = self.h[rand_hash](key)
            if new_slot_ind != evacuated_slot:  # we don't want to go back!
                break

        order = self.prg.n_choose_2(self.num_of_hashes)
        new_slot_ind = self.h[order[0]](key)
        if new_slot_ind == evacuated_slot:  # we don't want to go back!
            new_slot_ind = self.h[order[1]](key)

        [kicked_out_key, kicked_out_value] = self.buckets[new_slot_ind]
        self.buckets[new_slot_ind] = [key, value]
        return self._insert(kicked_out_key, kicked_out_value, new_slot_ind, path_len + 1)

    def __getitem__(self, key):
        hash_num = 0
        while hash_num < self.num_of_hashes:
            # go over all hash functions
            slot = self.h[hash_num](key)
            if self.buckets[slot][0] == key:
                return self.buckets[slot][1]
            hash_num += 1
        print("Error: key not found in Cuckoo hash!")

    def get(self, key, default):
        hash_num = 0
        while hash_num < self.num_of_hashes:
            # go over all hash functions
            slot = self.h[hash_num](key)
            if self.buckets[slot][0] == key:
                return self.buckets[slot][1]
            hash_num += 1
        return default

    def remove_item(self, key):
        hash_num = 0
        while hash_num < self.num_of_hashes:
            # go over all hash functions
            slot = self.h[hash_num](key)
            if self.buckets[slot][0] == key:
                self.buckets[slot] = [None, None]
                self.num_of_items -= 1
            hash_num += 1


class SegmentsQueue:
    """linked-list that preserves the order of insertion"""

    #                                                          tail (new segment insertion)
    #                                                           |
    #                    _____  next _____  next _____  next ___V_  next = null
    #                    | 0  |----->| 1  |----->| 2  |----->| 3  |----->
    #              <-----|____|<-----|____|<-----|____|<-----|____|
    #        prev = null   /|\  prev        prev        prev
    #                       |
    #                      head (remove_oldest)
    #

    def __init__(self, read_addresses, read_addresses_seg_dict_instance):
        self.tail = None  # dummy segment to initiate linked list.
        self.head = self.tail  # dummy segment to serve as head of segments linked list.
        self.read_addresses = read_addresses
        self.segments_count = 0
        self.read_addresses_seg_dict_instance = read_addresses_seg_dict_instance

    def remove_oldest(self):
        """Removes oldest segment from queue, *AND* from it's chunk, and returns it"""
        if self.head.slba_msbs in self.read_addresses:  # if not, the segment in head is a "wild" segment: it belonged to a chunk that was kicked out during a failed insertion
            chunk = self.read_addresses[self.head.slba_msbs]
            if len(chunk) == 1:
                if type(self.read_addresses) is dict:
                    del self.read_addresses[self.head.slba_msbs]
                else:
                    self.read_addresses.remove_item(self.head.slba_msbs)
            else:
                chunk.remove(self.head)  # remove from list in chunk
        res = self.head
        self.head = self.head.next
        self.head.prev = None
        self.segments_count -= 1
        self.read_addresses_seg_dict_instance.lbas_count -= res.nlb
        return res


class Segment:
    def __init__(self, slba_msbs, slba_lsbs, nlb, time_stamp_us, segment_queue_instance):

        self.queue = segment_queue_instance
        """Creates new segment and inserts it in tail of queue, but *NOT* to a chunk. This is handled separately."""
        self.slba_msbs = slba_msbs
        self.slba_lsbs = slba_lsbs
        self.nlb = nlb
        self.time_stamp_us = time_stamp_us
        if self.queue.segments_count == 0:
            self.queue.tail = self
            self.queue.head = self
            self.next = None
            self.prev = None
        else:
            self.prev = self.queue.tail
            self.queue.tail.next = self
            self.queue.tail = self
            self.next = None

        self.queue.segments_count += 1

    def remove(self):
        """Removes segment from queue, but *NOT* from the chunk. This is handled separately."""

        if self.queue.segments_count > 1:  # for C++ implementation: else: release single segment.
            if self.prev is None:  # segment is head
                self.queue.head = self.queue.head.next
                self.queue.head.prev = None
            else:
                self.prev.next = self.next

            if self.next is None:  # segment is tail
                self.queue.tail = self.queue.tail.prev
                self.queue.tail.next = None
            else:
                self.next.prev = self.prev
        self.queue.segments_count -= 1


class ReadAddressesSegmentsChunkedDict(ReadAddresses):
    def __init__(self, params):
        self.params = legacy_param_structure_handler(params)
        self.chunks_hash = CuckooHashTable(params)  # each entry saves all segments that start within some chunk.
        self.nlb_range = int(self.params["nlb_range"])
        self.chunk_size = self.nlb_range  # not obligatory. can be other values too.
        self.segments_queue = SegmentsQueue(self.chunks_hash, self)
        self.lbas_count = 0

    def probe_war_zone(self):
        y = [x for x in self.chunks_hash.buckets if x[0]]
        num_segments = np.array([len(z[1]) for z in y if z[1]])
        num_holes = [
            np.fromiter((z[1][i + 1].slba_lsbs - (z1.slba_lsbs + z1.nlb) > 0 for (i, z1) in enumerate(z[1][:-1])),
                        dtype=bool).sum() for z in y]
        nlb_segment_mean = np.array(
            [np.fromiter((z1.nlb for (i, z1) in enumerate(z[1][:])), dtype=int).mean() for z in y if z[1]])
        nlb_segment_std = np.array(
            [np.fromiter((z1.nlb for (i, z1) in enumerate(z[1][:])), dtype=int).std() for z in y if z[1]])
        nlb_hole_mean = np.array([np.mean(np.fromiter(
            ((z[1][i + 1].slba_lsbs - (z1.slba_lsbs + z1.nlb) > 0) * (z[1][i + 1].slba_lsbs - (z1.slba_lsbs + z1.nlb))
             for
             (i, z1) in enumerate(z[1][:-1])), dtype=int)) for z in y])
        nlb_hole_std = np.array([np.std(np.fromiter(
            ((z[1][i + 1].slba_lsbs - (z1.slba_lsbs + z1.nlb) > 0) * (z[1][i + 1].slba_lsbs - (z1.slba_lsbs + z1.nlb))
             for
             (i, z1) in enumerate(z[1][:-1])), dtype=int)) for z in y])

        return num_segments, num_holes, nlb_segment_mean, nlb_segment_std, nlb_hole_mean, nlb_hole_std

    def lbas_read(self, slba, nlb, time_stamp_us_r):

        slba_msbs = slba // self.chunk_size  # serves as key
        slba_lsbs = slba % self.chunk_size

        time_stamp_us = None
        if slba_msbs > 0:
            intersected_count, time_stamp_us = self.check_intersection_below(slba_msbs, slba_lsbs, nlb)
            contained_in_below = False  # maybe change so that if contained in below will be added to below
        else:
            contained_in_below = False
            intersected_count = 0
        if not contained_in_below:
            intersection, time_stamp_us_ = self.check_intersection_in_chunk(slba_msbs, slba_lsbs, nlb, time_stamp_us_r, True)
            if time_stamp_us is None:
                time_stamp_us = time_stamp_us_
            intersected_count += intersection
            if slba_lsbs + nlb > self.chunk_size:
                intersection, time_stamp_us_ = self.check_intersection_above(slba_msbs, slba_lsbs, nlb)
                if time_stamp_us is None:
                    time_stamp_us = time_stamp_us_
                intersected_count += intersection
        self.lbas_count += (nlb - intersected_count)
        self.chunks_hash.max_num_of_items = max(self.chunks_hash.max_num_of_items, self.chunks_hash.num_of_items)

        if time_stamp_us is not None:
            rar_lapse_command = time_stamp_us_r - time_stamp_us
        else:
            rar_lapse_command = None
        return intersected_count, rar_lapse_command

    def lbas_write(self, slba, nlb, time_stamp_us_w):

        slba_msbs = slba // self.chunk_size  # serves as key
        slba_lsbs = slba % self.chunk_size
        time_stamp_us = None
        if slba_msbs > 0:
            intersected_count, time_stamp_us = self.check_intersection_below(slba_msbs, slba_lsbs, nlb)
            contained_in_below = False  # maybe change so that if contained in below will be added to below
        else:
            contained_in_below = False
            intersected_count = 0
        if not contained_in_below:
            intersection, time_stamp_us_ = self.check_intersection_in_chunk(slba_msbs, slba_lsbs, nlb, time_stamp_us, False)
            if time_stamp_us is None:
                time_stamp_us = time_stamp_us_
            intersected_count += intersection
            if slba_lsbs + nlb > self.chunk_size:
                intersection, time_stamp_us_ = self.check_intersection_above(slba_msbs, slba_lsbs, nlb)
                if time_stamp_us is None:
                    time_stamp_us = time_stamp_us_
                intersected_count += intersection
        self.lbas_count -= intersected_count
        if time_stamp_us is not None:
            war_lapse_command = time_stamp_us_w - time_stamp_us
        else:
            war_lapse_command = None
        # read_nlb_command = -1 # for future implementation
        return intersected_count, war_lapse_command  # , read_nlb_command

    def get_lbas(self):
        print('warning: get_lbas not supported!')
        return []

    def get_size(self):
        return self.lbas_count

    def check_intersection_below(self, slba_msbs, slba_lsbs, nlb):
        if not slba_msbs - 1 in self.chunks_hash:
            return 0, None
        last_segment_below = self.chunks_hash[slba_msbs - 1][-1]
        gap = last_segment_below.slba_lsbs + last_segment_below.nlb - self.chunk_size - slba_lsbs
        intersection = 0
        timestamp = None
        if gap > 0:  # intersection with segment below
            if gap > nlb:  # new segment strictly contained in segment from chunk below
                # print("weirdness alert: new segment strictly contained in segment from chunk below!")

                # move suffix of segment from below one chunk up

                # connect it to 'last_segment_below' in FIFO (add prev and next pointers)
                right_suffix = Segment(slba_msbs, slba_lsbs + nlb, gap - nlb, last_segment_below.time_stamp_us, self.segments_queue)

                chunk = self.chunks_hash.get(slba_msbs, [])
                self.chunks_hash[slba_msbs] = [right_suffix] + chunk
                intersection = nlb
            else:
                intersection = gap
            last_segment_below.nlb -= gap  # remove intersection from below
            timestamp = last_segment_below.time_stamp_us
        return intersection, timestamp

    def check_intersection_in_chunk(self, slba_msbs, slba_lsbs, nlb, new_time_stamp_us, is_read):
        if not is_read and slba_msbs not in self.chunks_hash:
            return 0, None
        new_segment = Segment(slba_msbs, slba_lsbs, nlb, new_time_stamp_us, self.segments_queue) if is_read else None
        chunk = self.chunks_hash.get(slba_msbs, [])
        num_of_segments_to_remove = 0
        intersected_lbas_count = 0
        segment_num = 0
        time_stamp_us = None
        while segment_num < len(chunk):
            segment = chunk[segment_num]
            if segment.slba_lsbs + segment.nlb <= slba_lsbs:  # segment ends before new segment
                segment_num += 1
                continue
            if segment.slba_lsbs >= slba_lsbs + nlb:  # segment starts after new segment ends
                break
            time_stamp_us = segment.time_stamp_us
            start_gap = slba_lsbs - segment.slba_lsbs
            end_gap = (slba_lsbs + nlb) - (segment.slba_lsbs + segment.nlb)
            if end_gap >= 0:
                if start_gap > 0:  # new segment "bites" suffix of left segment
                    intersected_lbas_count += segment.nlb - start_gap
                    segment.nlb = start_gap
                else:  # new segment contains segment
                    intersected_lbas_count += segment.nlb
                    num_of_segments_to_remove += 1
                    # self.remove_segment(chunk, segment_num)
                    segment.remove()
            else:  # end_gap < 0
                if start_gap <= 0:  # new segment "bites" prefix of right segment
                    intersection = nlb + start_gap  # start_gap is non-positive
                    intersected_lbas_count += intersection
                    segment.slba_lsbs += intersection
                    segment.nlb -= intersection
                    if segment.slba_lsbs >= self.chunk_size:
                        # move segment one chunk up
                        segment.slba_msbs += 1
                        segment.slba_lsbs -= self.chunk_size
                        chunk_above = self.chunks_hash.get(slba_msbs + 1, [])
                        self.chunks_hash[slba_msbs + 1] = [segment] + chunk_above
                        if is_read:
                            new_chunk = chunk[:segment_num - num_of_segments_to_remove] + [new_segment]
                        else:
                            new_chunk = chunk[:segment_num - num_of_segments_to_remove]
                    else:
                        if is_read:
                            new_chunk = chunk[:segment_num - num_of_segments_to_remove] + [new_segment] + chunk[segment_num:]
                        else:
                            new_chunk = chunk[:segment_num - num_of_segments_to_remove] + chunk[segment_num:]

                    if len(new_chunk) > 0:
                        self.chunks_hash[slba_msbs] = new_chunk
                    else:
                        self.chunks_hash.remove_item(slba_msbs)
                    return intersected_lbas_count, time_stamp_us

                else:  # new segment bites segment in middle
                    intersected_lbas_count += nlb
                    segment.nlb = start_gap
                    # connect it to 'segment' in FIFO (add prev and next pointers)
                    right_suffix = Segment(slba_msbs, slba_lsbs + nlb, - end_gap, segment.time_stamp_us, self.segments_queue)

                    if right_suffix.slba_lsbs >= self.chunk_size:
                        # move right suffix one chunk up
                        right_suffix.slba_msbs += 1
                        right_suffix.slba_lsbs -= self.chunk_size
                        chunk_above = self.chunks_hash.get(slba_msbs + 1, [])
                        self.chunks_hash[slba_msbs + 1] = [right_suffix] + chunk_above

                        if is_read:
                            new_chunk = chunk[:segment_num + 1] + [new_segment]
                        else:
                            new_chunk = chunk[:segment_num + 1]
                    else:
                        if is_read:
                            new_chunk = chunk[:segment_num + 1] + [new_segment] + [right_suffix] + chunk[segment_num + 1:]
                        else:
                            new_chunk = chunk[:segment_num + 1] + [right_suffix] + chunk[segment_num + 1:]
                    self.chunks_hash[slba_msbs] = new_chunk
                    return intersected_lbas_count, time_stamp_us
            segment_num += 1
        if is_read:
            new_chunk = chunk[:segment_num - num_of_segments_to_remove] + [new_segment] + chunk[segment_num:]
            self.chunks_hash[slba_msbs] = new_chunk
        else:
            new_chunk = chunk[:segment_num - num_of_segments_to_remove] + chunk[segment_num:]
            if len(new_chunk) > 0:
                self.chunks_hash[slba_msbs] = new_chunk
            else:
                self.chunks_hash.remove_item(slba_msbs)

            # if len(chunk) == 0:
            #     del self.read_addresses[slba_msbs]
        if intersected_lbas_count < 0:  # debug
            print(str(intersected_lbas_count))
        return intersected_lbas_count, time_stamp_us

    def check_intersection_above(self, slba_msbs, slba_lsbs, nlb):
        # check how many segments of the (prefix of the) chunk above are intersected. remove all intersected, and keep only the non-intersected suffix of the last one.

        if not slba_msbs + 1 in self.chunks_hash:
            return 0, None
        chunk = self.chunks_hash[slba_msbs + 1]
        totally_contained_segments_count = 0
        aggregated_intersection = 0
        segment_num = 0
        time_stamp_us = None
        while segment_num < len(chunk):
            segment_above = chunk[segment_num]
            gap = slba_lsbs + nlb - self.chunk_size - segment_above.slba_lsbs  # it is not necessarily the intersection length; see below
            if gap > 0:
                intersection = min(gap, segment_above.nlb)
                time_stamp_us = segment_above.time_stamp_us
                if gap >= segment_above.nlb:  # new segment contains whole segment from chunk above
                    # if gap > segment_above.nlb:
                    #     print("weirdness alert: new segment strictly contains segment from chunk above!")
                    totally_contained_segments_count += 1
                    # self.remove_segment(chunk, segment_num)
                    segment_above.remove()
                else:
                    segment_above.nlb -= gap
                    segment_above.slba_lsbs += gap
                aggregated_intersection += intersection
            segment_num += 1
        chunk[:] = chunk[totally_contained_segments_count:]
        if len(chunk) == 0:
            self.chunks_hash.remove_item(slba_msbs + 1)

        return aggregated_intersection, time_stamp_us

    def balance_size(self):
        if self.params["balance_by"] == "segments":
            while self.segments_queue.segments_count > int(self.params['max_num_of_segments']) - 2:  # -2 because lbas_read can add 2 segments
                self.segments_queue.remove_oldest()
        elif self.params["balance_by"] == "lbas":
            while self.lbas_count > int(self.params["max_num_of_lbas"]):
                self.segments_queue.remove_oldest()

        elif self.params["balance_by"] == "none":
            return

        else:
            print('Segments queue size balancing can only be one of <segments, lbas, none>')


def legacy_param_structure_handler(params: dict) -> dict:
    if "preprocessor" in params and "RASCD" in params["preprocessor"]:
        params = params.copy()
        for k in params["preprocessor"]["RASCD"]:
            params[k] = params["preprocessor"]["RASCD"][k]
        params['balance_by'] = 'segments' if params["preprocessor"]["RASCD"]['balance_by_segments'] else 'lbas'
        del params["preprocessor"]
    return params
