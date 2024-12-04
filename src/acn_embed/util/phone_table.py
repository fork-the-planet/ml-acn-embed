def phone_pd_to_pi(phone):
    """
    Converts position-dependent phone to position-independent phone
    """
    # pylint: disable=too-many-boolean-expressions
    if (
        len(phone) > 2
        and phone[-2] == "_"
        and (phone[-1] == "S" or phone[-1] == "B" or phone[-1] == "I" or phone[-1] == "E")
    ):
        return phone[:-2]
    return phone


# pylint: disable=duplicate-code
class PhoneTable:
    def __init__(self, filename):
        self.id_to_pd_phone = {}
        self.pd_phone_to_id = {}
        with open(filename, "r", encoding="utf8") as fobj:
            for line in fobj:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                pd_phone = tokens[0]
                num = int(tokens[1])
                self.id_to_pd_phone[num] = pd_phone
                self.pd_phone_to_id[pd_phone] = num

    def get_pi_phone_set(self):
        pi_phone_set = set()
        for pd_phone in self.pd_phone_to_id:
            if pd_phone == "<eps>":
                continue
            if pd_phone.startswith("#"):
                continue
            pi_phone = phone_pd_to_pi(pd_phone)
            pi_phone_set.add(pi_phone)
        return pi_phone_set

    def write(self, filename):
        with open(filename, "w", encoding="utf8") as fobj:
            for phone_id in sorted(self.id_to_pd_phone.keys()):
                fobj.write(f"{self.id_to_pd_phone[phone_id]} {phone_id}\n")
