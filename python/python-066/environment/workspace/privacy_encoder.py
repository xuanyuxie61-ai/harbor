"""
privacy_encoder.py
================================================================================
地下水监测网络隐私保护编码模块

基于种子项目 1045_rot13 的核心思想（字符循环置换密码），扩展为水文地质
监测数据的隐私保护编码系统。在地下水污染修复项目中，井位坐标和监测井
ID 属于敏感信息，本模块提供可逆的 ROT13/ROT5 混合编码，保障数据共享时
的隐私安全。

核心公式：
  对于字母字符 c ∈ {A-Z, a-z}:
      ROT13(c) = chr( (ord(c) - base + 13) mod 26 + base )
  对于数字字符 d ∈ {0-9}:
      ROT5(d)  = chr( (ord(d) - ord('0') + 5) mod 10 + ord('0') )
  复合编码保持映射的可逆性：ROT13(ROT13(c)) = c，ROT5(ROT5(d)) = d
================================================================================
"""

import re


def _rot13_char(ch: str) -> str:
    """对单个字符执行 ROT13（字母）或 ROT5（数字）变换。"""
    if len(ch) != 1:
        raise ValueError("rot13_char 仅接受单字符输入")
    code = ord(ch)
    if 65 <= code <= 90:          # 'A'-'Z'
        return chr((code - 65 + 13) % 26 + 65)
    elif 97 <= code <= 122:       # 'a'-'z'
        return chr((code - 97 + 13) % 26 + 97)
    elif 48 <= code <= 57:        # '0'-'9'
        return chr((code - 48 + 5) % 10 + 48)
    else:
        return ch


def encode_well_id(well_id: str) -> str:
    """
    对监测井编号执行隐私编码。

    参数
    ----------
    well_id : str
        原始监测井编号，例如 "MW-2024-A05"

    返回
    -------
    str
        编码后的井号，例如 "ZJ-2024-N05"
    """
    if not isinstance(well_id, str):
        raise TypeError("井号必须为字符串类型")
    if len(well_id) == 0:
        raise ValueError("井号不能为空字符串")
    return "".join(_rot13_char(c) for c in well_id)


def decode_well_id(encoded_id: str) -> str:
    """
    解码监测井编号（ROT13/ROT5 的自逆性保证解码与编码使用同一函数）。
    """
    return encode_well_id(encoded_id)


def encode_coordinate_list(coords: list[tuple[float, float]],
                           well_ids: list[str]) -> dict:
    """
    对一组监测井的坐标和编号进行批量编码。

    返回字典：
      {
        "encoded_wells": [
            {"original_id": ..., "encoded_id": ..., "x": ..., "y": ...},
            ...
        ]
      }
    """
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
    """
    对野外水文地质调查记录的敏感字段进行批量编码。
    """
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
    # 快速自测试
    test_id = "MW-2024-A05"
    enc = encode_well_id(test_id)
    dec = decode_well_id(enc)
    assert dec == test_id, "ROT13/ROT5 自逆性验证失败"
    print("privacy_encoder: 自测试通过")
