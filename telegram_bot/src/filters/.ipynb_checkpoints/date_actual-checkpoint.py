from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re

WORD_TO_NUM = {
    'один': 1, 'одну': 1, 'одной': 1, 'раз': 1,
    'два': 2, 'две': 2, 'три': 3, 'четыре': 4, 
    'пять': 5, 'шесть': 6, 'семь': 7, 'восемь': 8, 
    'девять': 9, 'десять': 10
}

MONTHS = {
    'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
    'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
    'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
}

TIME_CONFIG = [
    {
        'keywords': ['день', 'дня', 'дней', 'дню'],
        'units': 'days',
        'delta': lambda x: timedelta(days=x)
    },
    {
        'keywords': ['недел', 'неделю', 'неделями','недели'],
        'units': 'weeks',
        'delta': lambda x: timedelta(weeks=x)
    },
    {
        'keywords': ['месяц', 'месяца', 'месяцев', 'месяцу'],
        'units': 'months',
        'delta': lambda x: relativedelta(months=x)
    },
    {
        'keywords': ['год', 'года', 'лет', 'году'],
        'units': 'years',
        'delta': lambda x: relativedelta(years=x)
    }
]


def extract_dates(
    text: str,
    base_date: datetime = None,
    is_query: bool = False,
    strict_mode: bool = False,
    month_threshold: int = 2,
    day_threshold: int = 15,
) -> list[str]:
    '''
    Универсальная функция извлечения дат из текста
    
    Параметры:
    - text: текст для анализа
    - base_date: базовая дата для относительных вычислений ("сегодня" для определенного чанка)
    - is_query: True для пользовательских запросов (более гибкий парсинг)
    - strict_mode: True для точного соответствия форматам (меньше false positives)
    - month_threshold: порог месяцев для определения смены года
    - day_threshold: порог дней для определения смены года
    
    Возвращает ['dd.mm.YYYY']
    '''
    if base_date is None:
        base_date = datetime.now()
    
    normalized_text = _normalize_text(text, is_query)
    processed_text = replace_temporal_expressions(normalized_text, base_date)
    
    # Сначала ищем полные даты
    full_dates = _extract_full_dates(processed_text)
    
    # Затем неполные даты (если не strict_mode)
    partial_dates = []
    if not strict_mode:
        partial_dates = _extract_partial_dates(
            processed_text, 
            base_date,
            month_threshold,
            day_threshold,
            exclude_positions=[(start, end) for start, end, _ in full_dates]
        )
    
    all_dates = [date for _, _, date in full_dates] + partial_dates
    
    # Добавляем специальные случаи для запросов
    if is_query:
        all_dates.extend(_process_query_special_cases(text, base_date))
    
    return sorted(list(set(all_dates)))


def _extract_full_dates(text: str) -> list[tuple]:
    '''
    Извлекает полные даты с их позициями в тексте
    '''
    patterns = [
        # Полные даты с разными разделителями
        r'(?<!\d)(0?[1-9]|[12][0-9]|3[01])[./-](0?[1-9]|1[0-2])[./-](\d{4})(?!\d)',
        r'(?<!\d)(0?[1-9]|[12][0-9]|3[01])\s+(0?[1-9]|1[0-2])\s+(\d{4})(?!\d)',
        r'(?<!\d)(0?[1-9]|[12][0-9]|3[01])\s+([а-я]+)\s+(\d{4})(?!\d)',
        r'(?<!\d)(0?[1-9]|[12][0-9]|3[01])\s+([а-я]+)\s+(\d{4})\s*года(?!\d)',
    ]
    
    dates = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            day, month, year = match.groups()
            if not month.isdigit():
                month = _month_to_num(month)
                if not month:
                    continue
            
            if _is_valid_date(int(day), int(month), int(year)):
                date_str = f"{int(day):02d}.{int(month):02d}.{year}"
                dates.append((match.start(), match.end(), date_str))
    
    return dates


def _extract_partial_dates(
    text: str,
    base_date: datetime,
    month_threshold: int,
    day_threshold: int,
    exclude_positions: list[tuple]
) -> list[str]:
    '''
    Извлекает неполные даты, исключая перекрытия с полными
    '''
    patterns = [
        r'(?<!\d)(0?[1-9]|[12][0-9]|3[01])[./-](0?[1-9]|1[0-2])(?![./-]\d)', # dd.mm
        r'(?<!\d)(0?[1-9]|[12][0-9]|3[01])\s+([а-я]+)(?!\s+\d)', # dd месяц
        r'(?<!\d)(0?[1-9]|[12][0-9]|3[01])\s+(?:числа|число)(?!\d)', # dd числа
    ]
    
    dates = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Проверяем, не входит ли в уже найденные полные даты
            if any(start <= match.start() < end for start, end in exclude_positions):
                continue
            
            groups = match.groups()
            if len(groups) == 2:
                day, month = groups
                if not month.isdigit():
                    month = _month_to_num(month)
                    if not month:
                        continue
                
                year = _determine_year_with_boundary(
                    int(day), int(month), base_date, month_threshold, day_threshold
                )
                date_str = f"{int(day):02d}.{int(month):02d}.{year}"
                dates.append(date_str)
                
            elif len(groups) == 1:
                day = int(groups[0])
                month = base_date.month
                year = base_date.year
                if not _is_valid_date(day, month, year):
                    month = month % 12 + 1
                    year = year if month > 1 else year + 1
                date_str = f"{day:02d}.{month:02d}.{year}"
                dates.append(date_str)
    
    return dates              


def _normalize_text(text: str, is_query: bool) -> str:
    '''
    Некоторая "нормализация" текста (удаление лишних символов, замена n-го на n)
    '''
    text = text.lower()
    if is_query:
        text = re.sub(r'[?¿]', '', text)
        text = re.sub(r'\b(\d{1,2})(?:-го|-е)\b', r'\1', text)
    return text


def replace_temporal_expressions(text: str, today: datetime) -> str:
    text = _replace_base_keywords(text, today)
    text = _replace_intervals(text, today)
    return text


def _replace_base_keywords(text: str, today: datetime) -> str:
    '''
    Замена базовых выражений (сегодня, вчера и т.д.)
    '''
    replacements = {
        r'\bсегодня\b': today,
        r'\bсейчас\b': today,
        r'\bвчера\b': today - timedelta(days=1),
        r'\bзавтра\b': today + timedelta(days=1),
        r'\bактуал(?:ьный|ьная|ьное|ьные|ьны|ьна|ен|но)\b': today
    }
    
    for pattern, date in replacements.items():
        text = re.sub(pattern, date.strftime('%d.%m.%Y'), text, flags=re.IGNORECASE)
    
    return text


def _replace_intervals(text: str, today: datetime) -> str:
    '''
    Замена интервалов (3 дня назад, через 2 недели и т.д.)
    '''
    for config in TIME_CONFIG:
        # Формируем паттерны для "назад" и "через"
        num_pattern = f"(\\d+|{'|'.join(WORD_TO_NUM.keys())})"
        keyword_pattern = f"({'|'.join(config['keywords'])})"
        
        # Паттерн для "назад"
        pattern_back = f"(?:{num_pattern}\\s+)?{keyword_pattern}\\s+(назад|тому назад)"
        text = re.sub(
            pattern_back,
            lambda m: (today - config['delta'](_parse_number(m))).strftime('%d.%m.%Y'),
            text,
            flags=re.IGNORECASE
        )        
        # Паттерн для "через"
        pattern_forward = f"через\\s+(?:{num_pattern}\\s+)?{keyword_pattern}"
        text = re.sub(
            pattern_forward,
            lambda m: (today + config['delta'](_parse_number(m))).strftime('%d.%m.%Y'),
            text,
            flags=re.IGNORECASE
        )
    
    return text


def _month_to_num(month: str) -> int:
    '''
    Преобразование названия месяца в число
    '''
    month = month.lower()

    return MONTHS.get(month, None)


def _parse_number(match) -> int:
    '''
    Извлечение числа из текста (цифра или слово)
    '''
    num_str = match.group(1) or '1'  # Если число не указано, по умолчанию 1
    return int(num_str) if num_str.isdigit() else WORD_TO_NUM.get(num_str.lower(), 1)
    

def _is_valid_date(day: int, month: int, year: int) -> bool:
    try:
        datetime(year=year, month=month, day=day)
        return True
    except ValueError:
        return False


def _parse_text_month_date(
    groups: list,
    base_date: datetime,
    month_threshold: int,
    day_threshold: int
) -> list[str]:
    '''
    Обработка дат с текстовым месяцем
    '''
    day = int(groups[0])
    month = _month_to_num(groups[1])
    year = int(groups[2]) if len(groups) > 2 else base_date.year
    
    # Если год не указан, определяем с учетом границ года
    if len(groups) == 2:
        year = _determine_year_with_boundary(
            day=day,
            month=month,
            base_date=base_date,
            month_threshold=month_threshold,
            day_threshold=day_threshold
        )
    
    return [f"{day:02d}.{month:02d}.{year}"]


def _parse_partial_date(
    groups: list,
    base_date: datetime,
    month_threshold: int,
    day_threshold: int
) -> list[str]:
    '''
    Обработка неполных дат (dd.mm)
    '''
    day, month = map(int, groups[:2])
    year = _determine_year_with_boundary(
        day=day,
        month=month,
        base_date=base_date,
        month_threshold=month_threshold,
        day_threshold=day_threshold
    )
    return [f"{day:02d}.{month:02d}.{year}"]



def _parse_day_only(
    day_str: str,
    base_date: datetime,
    month_threshold: int,
    day_threshold: int
) -> list[str]:
    '''
    Обработка указания только дня
    '''
    day = int(day_str)
    month = base_date.month
    year = base_date.year
    
    if not _is_valid_date(day, month, year):
        next_month = base_date.month % 12 + 1
        next_year = base_date.year if next_month > 1 else base_date.year + 1
        if _is_valid_date(day, next_month, next_year):
            return [f"{day:02d}.{next_month:02d}.{next_year}"]
        return []
    
    # Проверяем, не находится ли дата близко к границе года
    year = _determine_year_with_boundary(
        day=day,
        month=month,
        base_date=base_date,
        month_threshold=month_threshold,
        day_threshold=day_threshold
    )
    
    return [f"{day:02d}.{month:02d}.{year}"]


def _determine_year_with_boundary(
    day: int,
    month: int,
    base_date: datetime,
    month_threshold: int,
    day_threshold: int
) -> int:
    """Определяет год для неполных дат с учетом границ года"""
    base_month = base_date.month
    base_day = base_date.day
    base_year = base_date.year
    
    # Специальная обработка декабря и января
    if base_month == 12 and month == 1:
        return base_year + 1 if day <= day_threshold else base_year
    if base_month == 1 and month == 12:
        return base_year - 1 if day > day_threshold else base_year
    
    # Для остальных месяцев
    month_diff = (month - base_month) % 12
    if month_diff <= month_threshold or (12 - month_diff) <= month_threshold:
        if (month > base_month) or (month == base_month and day > base_day):
            return base_year
        return base_year + 1 if base_month == 12 and month == 1 else base_year
    
    return base_year


def _process_query_special_cases(text: str, base_date: datetime) -> list[str]:
    '''
    Обработка специальных случаев для запросов
    '''
    cases = []
    
    # Обработка диапазонов (14-15 числа)
    range_match = re.search(r'(\d{1,2})-(\d{1,2})\s+числа', text)
    if range_match:
        day1, day2 = map(int, range_match.groups())
        month = base_date.month
        year = base_date.year
        cases.extend(f"{d:02d}.{month:02d}.{year}" for d in range(day1, day2+1))
    
    return cases


def _determine_year(day: int, month: int, base_date: datetime) -> int:
    '''
    Определение года для неполных дат
    '''
    current_year = base_date.year
    if (month > base_date.month) or (month == base_date.month and day > base_date.day):
        return current_year
    return current_year

