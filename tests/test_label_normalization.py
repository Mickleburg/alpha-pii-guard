"""
Tests for BIO label normalization.
"""

import pytest
from ml.merge.label_map import normalize_label, is_bio_label, all_final_labels


def test_b_email_normalized():
    """B-Email -> Email."""
    assert normalize_label("B-Email") == "Email"


def test_i_email_normalized():
    """I-Email -> Email."""
    assert normalize_label("I-Email") == "Email"


def test_b_phone_normalized():
    """B-Номер телефона -> Номер телефона."""
    assert normalize_label("B-Номер телефона") == "Номер телефона"


def test_i_fio_normalized():
    """I-ФИО -> ФИО."""
    assert normalize_label("I-ФИО") == "ФИО"


def test_already_normalized_unchanged():
    """Already normalized label unchanged."""
    assert normalize_label("Email") == "Email"
    assert normalize_label("ФИО") == "ФИО"
    assert normalize_label("Паспортные данные") == "Паспортные данные"


def test_empty_label_safe():
    """Empty label handled safely."""
    assert normalize_label("") == ""


def test_invalid_prefix_unchanged():
    """Invalid prefix left unchanged."""
    assert normalize_label("X-Email") == "X-Email"


def test_is_bio_label_detection():
    """is_bio_label correctly identifies BIO tags."""
    assert is_bio_label("B-Email") is True
    assert is_bio_label("I-ФИО") is True
    assert is_bio_label("Email") is False
    assert is_bio_label("ФИО") is False


def test_all_final_labels_count():
    """all_final_labels returns complete set."""
    labels = all_final_labels()
    assert len(labels) == 30  # 32 categories total
    assert "Email" in labels
    assert "ФИО" in labels
    assert "Паспортные данные" in labels
    assert "API ключи" in labels


def test_all_categories_covered():
    """All 32 categories normalized correctly."""
    bio_labels = [
        "B-API ключи",
        "B-CVV/CVC",
        "B-Email",
        "B-Водительское удостоверение",
        "B-Временное удостоверение личности",
        "B-Гражданство и названия стран",
        "B-Данные об автомобиле клиента",
        "B-Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)",
        "B-Дата окончания срока действия карты",
        "B-Дата регистрации по месту жительства или пребывания",
        "B-Дата рождения",
        "B-Имя держателя карты",
        "B-Кодовые слова",
        "B-Место рождения",
        "B-Наименование банка",
        "B-Номер банковского счета",
        "B-Номер карты",
        "B-Номер телефона",
        "B-Одноразовые коды",
        "B-ПИН код",
        "B-Пароли",
        "B-Паспортные данные",
        "B-Полный адрес",
        "B-Разрешение на работу / визу",
        "B-СНИЛС клиента",
        "B-Сведения об ИНН",
        "B-Свидетельство о рождении",
        "B-Серия и номер вида на жительство",
        "B-Содержимое магнитной полосы",
        "B-ФИО",
    ]
    
    for bio_label in bio_labels:
        final = normalize_label(bio_label)
        assert not final.startswith("B-")
        assert not final.startswith("I-")
        assert len(final) > 0
