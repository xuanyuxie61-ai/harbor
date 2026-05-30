
import re


def _rot13_char(ch: str) -> str:
    if len(ch) != 1:
        raise ValueError("rot13_char 仅接受单字符输入")
    code = ord(ch)
    if 65 <= code <= 90:
        return chr((code - 65 + 13) % 26 + 65)
    elif 97 <= code <= 122:
        return chr((code - 97 + 13) % 26 + 97)
    elif 48 <= code <= 57:
        return chr((code - 48 + 5) % 10 + 48)
    else:
        return ch


def encode_well_id(well_id: str) -> str:
    if not isinstance(well_id, str):
        raise TypeError("井号必须为字符串类型")
    if len(well_id) == 0:
        raise ValueError("井号不能为空字符串")
    return "".join(_rot13_char(c) for c in well_id)


def decode_well_id(encoded_id: str) -> str:
    return encode_well_id(encoded_id)


def encode_coordinate_list(coords: list[tuple[float, float]],
                           well_ids: list[str]) -> dict:
    if len(coords) != len(well_ids):
        raise ValueError("坐标列表与井号列表长度必须一致")
    result = []
    for (x, y), wid in zip(coords, well_ids):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("坐标必须为数值类型")
        if not (-1e6 <= x <= 1e6 and -1e6 <= y <= 1e6):
            raise ValueError("坐标值超出合理范围")
        result.append({
            "original_id": wid,
            "encoded_id": encode_well_id(wid),
            "x": round(float(x), 6),
            "y": round(float(y), 6)
        })
    return {"encoded_wells": result}


def batch_encode_field_data(field_records: list[dict]) -> list[dict]:
    encoded = []
    for rec in field_records:
        new_rec = rec.copy()
        if "well_id" in new_rec:
            new_rec["well_id"] = encode_well_id(new_rec["well_id"])
        if "operator" in new_rec:
            new_rec["operator"] = encode_well_id(new_rec["operator"])
        encoded.append(new_rec)
    return encoded


if __name__ == "__main__":

    test_id = "MW-2024-A05"
    enc = encode_well_id(test_id)
    dec = decode_well_id(enc)
    assert dec == test_id, "ROT13/ROT5 自逆性验证失败"
    print("privacy_encoder: 自测试通过")
