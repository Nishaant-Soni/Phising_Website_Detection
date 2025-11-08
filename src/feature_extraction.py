"""
Feature Extraction Pipeline for Phishing Website Detection
Extracts 30 features from a URL to detect phishing websites
"""

import ipaddress
import re
import urllib.request
from bs4 import BeautifulSoup
import socket
import requests
import whois
from datetime import date, datetime
import time
from dateutil.parser import parse as date_parse
from urllib.parse import urlparse
from googlesearch import search

class FeatureExtraction:
    features = []
    def __init__(self,url):
        self.features = []
        self.url = url
        self.domain = ""
        self.whois_response = ""
        self.urlparse = ""
        self.response = ""
        self.soup = ""

        try:
            self.response = requests.get(url)
            self.soup = BeautifulSoup(self.response.text, 'html.parser')
        except:
            pass

        try:
            self.urlparse = urlparse(url)
            self.domain = self.urlparse.netloc
        except:
            pass

        try:
            self.whois_response = whois.whois(self.domain)
        except:
            pass

        self.features.append(self.UsingIp())              # having_IP_Address
        self.features.append(self.longUrl())              # URL_Length
        self.features.append(self.shortUrl())             # Shortining_Service
        self.features.append(self.symbol())               # having_At_Symbol
        self.features.append(self.redirecting())          # double_slash_redirecting
        self.features.append(self.prefixSuffix())         # Prefix_Suffix
        self.features.append(self.SubDomains())           # having_Sub_Domain
        self.features.append(self.Https())                # SSLfinal_State
        self.features.append(self.DomainRegLen())         # Domain_registeration_length
        self.features.append(self.Favicon())              # Favicon
        
        self.features.append(self.NonStdPort())           # port
        self.features.append(self.HTTPSDomainURL())       # HTTPS_token
        self.features.append(self.RequestURL())           # Request_URL
        self.features.append(self.AnchorURL())            # URL_of_Anchor
        self.features.append(self.LinksInScriptTags())    # Links_in_tags
        self.features.append(self.ServerFormHandler())    # SFH
        self.features.append(self.InfoEmail())            # Submitting_to_email
        self.features.append(self.AbnormalURL())          # Abnormal_URL
        self.features.append(self.WebsiteForwarding())    # Redirect
        self.features.append(self.StatusBarCust())        # on_mouseover

        self.features.append(self.DisableRightClick())    # RightClick
        self.features.append(self.UsingPopupWindow())     # popUpWidnow
        self.features.append(self.IframeRedirection())    # Iframe
        self.features.append(self.AgeofDomain())          # age_of_domain
        self.features.append(self.DNSRecording())         # DNSRecord
        self.features.append(self.WebsiteTraffic())       # web_traffic
        self.features.append(self.PageRank())             # Page_Rank
        self.features.append(self.GoogleIndex())          # Google_Index
        self.features.append(self.LinksPointingToPage())  # Links_pointing_to_page
        self.features.append(self.StatsReport())          # Statistical_report

    # 1.UsingIp - having_IP_Address
    def UsingIp(self):
        try:
            host = self.domain
            # strip port if present
            if ':' in host:
                host = host.split(':', 1)[0]
            # IPv6 may be wrapped in [ ]
            host = host.strip('[]')
            ipaddress.ip_address(host)
            return -1
        except:
            return 1

    # 2.longUrl - URL_Length
    def longUrl(self):
        if len(self.url) < 54:
            return 1
        if len(self.url) >= 54 and len(self.url) <= 75:
            return 0
        return -1

    # 3.shortUrl - Shortining_Service
    def shortUrl(self):
        match = re.search(
            r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
            r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
            r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
            r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
            r'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
            r'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
            r'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net',
            self.url,
        )
        if match:
            return -1
        return 1

    # 4.Symbol@ - having_At_Symbol
    def symbol(self):
        if re.findall("@",self.url):
            return -1
        return 1
    
    # 5.Redirecting// - double_slash_redirecting
    def redirecting(self):
        if self.url.rfind('//')>6:
            return -1
        return 1
    
    # 6.prefixSuffix - Prefix_Suffix
    def prefixSuffix(self):
        try:
            match = re.findall(r'\-', self.domain)
            if match:
                return -1
            return 1
        except:
            return -1
    
    # 7.SubDomains - having_Sub_Domain
    def SubDomains(self):
        # Count dots only in the host (not in path/query)
        host = self.domain
        if ':' in host:
            host = host.split(':', 1)[0]
        host = host.lower()
        # Normalize common prefix
        if host.startswith('www.'):
            host = host[4:]
        dot_count = host.count('.')
        if dot_count == 1:
            return 1
        elif dot_count == 2:
            return 0
        return -1

    # 8.HTTPS - SSLfinal_State
    def Https(self):
        try:
            https = self.urlparse.scheme
            if 'https' in https:
                return 1
            return -1
        except:
            return 1

    # 9.DomainRegLen - Domain_registeration_length
    def DomainRegLen(self):
        try:
            expiration_date = self.whois_response.expiration_date
            creation_date = self.whois_response.creation_date
            try:
                if(len(expiration_date)):
                    expiration_date = expiration_date[0]
            except:
                pass
            try:
                if(len(creation_date)):
                    creation_date = creation_date[0]
            except:
                pass

            age = (expiration_date.year-creation_date.year)*12+ (expiration_date.month-creation_date.month)
            if age >=12:
                return 1
            return -1
        except:
            return -1

    # 10. Favicon
    def Favicon(self):
        try:
            for head in self.soup.find_all('head'):
                for head.link in self.soup.find_all('link', href=True):
                    dots = [x.start(0) for x in re.finditer(r'\.', head.link['href'])]
                    if self.url in head.link['href'] or len(dots) == 1 or self.domain in head.link['href']:
                        return 1
            return -1
        except:
            return -1

    # 11. NonStdPort - port
    def NonStdPort(self):
        try:
            port = self.domain.split(":")
            if len(port)>1:
                return -1
            return 1
        except:
            return -1

    # 12. HTTPSDomainURL - HTTPS_token
    def HTTPSDomainURL(self):
        try:
            if 'https' in self.domain:
                return -1
            return 1
        except:
            return -1
    
    # 13. RequestURL - Request_URL
    def RequestURL(self):
        try:
            total = 0
            internal = 0

            def is_external(src: str) -> bool:
                if not src:
                    return False
                parsed = urlparse(src)
                if not parsed.netloc:
                    # relative path => internal
                    return False
                # Compare hostnames (strip ports)
                a = parsed.netloc.split(':', 1)[0].lower()
                b = self.domain.split(':', 1)[0].lower()
                return a != b

            for tag in self.soup.find_all(['img', 'audio', 'embed', 'iframe']):
                src = tag.get('src')
                if src is None:
                    continue
                total += 1
                if not is_external(src):
                    internal += 1

            if total == 0:
                return 1
            percentage_internal = (internal / float(total)) * 100
            if percentage_internal >= 78.0:  # <22% external
                return 1
            elif percentage_internal >= 39.0:  # 22–61% external
                return 0
            else:
                return -1
        except:
            return 0
    
    # 14. AnchorURL - URL_of_Anchor
    def AnchorURL(self):
        try:
            total = 0
            unsafe = 0

            def is_external(href: str) -> bool:
                if not href:
                    return True
                if href.startswith('#'):
                    return True
                low = href.lower()
                if low.startswith('javascript') or low.startswith('mailto:'):
                    return True
                parsed = urlparse(href)
                if not parsed.netloc:
                    return False  # relative link => internal/safe
                a = parsed.netloc.split(':', 1)[0].lower()
                b = self.domain.split(':', 1)[0].lower()
                return a != b

            for a in self.soup.find_all('a', href=True):
                total += 1
                if is_external(a['href']):
                    unsafe += 1

            if total == 0:
                return 1
            pct_unsafe = (unsafe / float(total)) * 100
            if pct_unsafe < 31.0:
                return 1
            elif pct_unsafe < 67.0:
                return 0
            else:
                return -1
        except:
            return 0

    # 15. LinksInScriptTags - Links_in_tags
    def LinksInScriptTags(self):
        try:
            total = 0
            internal = 0

            def is_external(urlish: str) -> bool:
                if not urlish:
                    return False
                parsed = urlparse(urlish)
                if not parsed.netloc:
                    return False
                a = parsed.netloc.split(':', 1)[0].lower()
                b = self.domain.split(':', 1)[0].lower()
                return a != b

            for link in self.soup.find_all('link', href=True):
                total += 1
                if not is_external(link['href']):
                    internal += 1
            for script in self.soup.find_all('script', src=True):
                total += 1
                if not is_external(script['src']):
                    internal += 1

            if total == 0:
                return 1
            pct_internal = (internal / float(total)) * 100
            if pct_internal >= 83.0:  # <17% external
                return 1
            elif pct_internal >= 19.0:  # 17–81% external
                return 0
            else:
                return -1
        except:
            return 0

    # 16. ServerFormHandler - SFH
    def ServerFormHandler(self):
        try:
            if len(self.soup.find_all('form', action=True))==0:
                return 1
            else :
                for form in self.soup.find_all('form', action=True):
                    if form['action'] == "" or form['action'] == "about:blank":
                        return -1
                    elif self.url not in form['action'] and self.domain not in form['action']:
                        return 0
                    else:
                        return 1
        except:
            return -1

    # 17. InfoEmail - Submitting_to_email
    def InfoEmail(self):
        try:
            if re.findall(r"[mail\(\)|mailto:?]", str(self.soup)):
                return -1
            else:
                return 1
        except:
            return -1

    # 18. AbnormalURL - Abnormal_URL
    def AbnormalURL(self):
        try:
            info = self.whois_response
            if not info:
                return -1
            domain_name = getattr(info, 'domain_name', None)
            if isinstance(domain_name, list):
                domain_name = domain_name[0] if domain_name else None
            return 1 if domain_name else -1
        except:
            return -1

    # 19. WebsiteForwarding - Redirect
    def WebsiteForwarding(self):
        try:
            if len(self.response.history) <= 1:
                return 1
            elif len(self.response.history) <= 4:
                return 0
            else:
                return -1
        except:
             return -1

    # 20. StatusBarCust - on_mouseover
    def StatusBarCust(self):
        try:
            if re.findall("<script>.+onmouseover.+</script>", self.response.text):
                return -1
            else:
                return 1
        except:
             return 1

    # 21. DisableRightClick - RightClick
    def DisableRightClick(self):
        try:
            if re.findall(r"event.button ?== ?2", self.response.text):
                return -1
            else:
                return 1
        except:
             return 1

    # 22. UsingPopupWindow - popUpWidnow
    def UsingPopupWindow(self):
        try:
            if re.findall(r"alert\(", self.response.text):
                return -1
            else:
                return 1
        except:
             return 1

    # 23. IframeRedirection - Iframe
    def IframeRedirection(self):
        try:
            # Presence of iframe element is considered suspicious in this feature set
            return -1 if self.soup and self.soup.find('iframe') else 1
        except:
            return 1

    # 24. AgeofDomain - age_of_domain
    def AgeofDomain(self):
        try:
            creation_date = self.whois_response.creation_date
            try:
                if(len(creation_date)):
                    creation_date = creation_date[0]
            except:
                pass

            today  = date.today()
            age = (today.year-creation_date.year)*12+(today.month-creation_date.month)
            if age >=6:
                return 1
            return -1
        except:
            return -1

    # 25. DNSRecording - DNSRecord
    def DNSRecording(self):
        try:
            # Resolve DNS A record for the domain
            host = self.domain.split(':', 1)[0]
            socket.gethostbyname(host)
            return 1
        except:
            return -1

    # 26. WebsiteTraffic - web_traffic
    def WebsiteTraffic(self):
        try:
            # Alexa API is deprecated. Default to neutral when unknown.
            return 0
        except:
            return 0

    # 27. PageRank - Page_Rank
    def PageRank(self):
        try:
            # Google PageRank is deprecated. Default to neutral when unknown.
            return 0
        except:
            return 0
            

    # 28. GoogleIndex - Google_Index
    def GoogleIndex(self):
        try:
            site = search(self.url, 5)
            if site:
                return 1
            else:
                return -1
        except:
            return 1

    # 29. LinksPointingToPage - Links_pointing_to_page
    def LinksPointingToPage(self):
        try:
            number_of_links = len(re.findall(r"<a href=", self.response.text))
            if number_of_links == 0:
                return -1  # No links = suspicious
            elif number_of_links <= 2:
                return 0   # Few links = borderline
            else:
                return 1   # Many links = legitimate (normal for real sites)
        except:
            return -1

    # 30. StatsReport - Statistical_report
    def StatsReport(self):
        try:
            url_match = re.search(
        r'at\.ua|usa\.cc|baltazarpresentes\.com\.br|pe\.hu|esy\.es|hol\.es|sweddy\.com|myjino\.ru|96\.lt|ow\.ly', self.url)
            ip_address = socket.gethostbyname(self.domain)
            ip_match = re.search(r'146\.112\.61\.108|213\.174\.157\.151|121\.50\.168\.88|192\.185\.217\.116|78\.46\.211\.158|181\.174\.165\.13|46\.242\.145\.103|121\.50\.168\.40|83\.125\.22\.219|46\.242\.145\.98|'
                                r'107\.151\.148\.44|107\.151\.148\.107|64\.70\.19\.203|199\.184\.144\.27|107\.151\.148\.108|107\.151\.148\.109|119\.28\.52\.61|54\.83\.43\.69|52\.69\.166\.231|216\.58\.192\.225|'
                                r'118\.184\.25\.86|67\.208\.74\.71|23\.253\.126\.58|104\.239\.157\.210|175\.126\.123\.219|141\.8\.224\.221|10\.10\.10\.10|43\.229\.108\.32|103\.232\.215\.140|69\.172\.201\.153|'
                                r'216\.218\.185\.162|54\.225\.104\.146|103\.243\.24\.98|199\.59\.243\.120|31\.170\.160\.61|213\.19\.128\.77|62\.113\.226\.131|208\.100\.26\.234|195\.16\.127\.102|195\.16\.127\.157|'
                                r'34\.196\.13\.28|103\.224\.212\.222|172\.217\.4\.225|54\.72\.9\.51|192\.64\.147\.141|198\.200\.56\.183|23\.253\.164\.103|52\.48\.191\.26|52\.214\.197\.72|87\.98\.255\.18|209\.99\.17\.27|'
                                r'216\.38\.62\.18|104\.130\.124\.96|47\.89\.58\.141|78\.46\.211\.158|54\.86\.225\.156|54\.82\.156\.19|37\.157\.192\.102|204\.11\.56\.48|110\.34\.231\.42', ip_address)
            if url_match:
                return -1
            elif ip_match:
                return -1
            return 1
        except:
            return 1
    
    def getFeaturesList(self):
        return self.features


def extract_features_from_url(url):
    """
    Main function to extract features from a URL
    Args:
        url: Website URL to analyze
    Returns:
        List of features matching dataset column order
    """
    extractor = FeatureExtraction(url)
    features = extractor.getFeaturesList()
    return features


if __name__ == "__main__":
    # Example usage
    url = 'https://www.google.com'
    print(f"Extracting features from: {url}")
    
    # Get features as list (matches dataset order)
    features = extract_features_from_url(url)
    print(f"\nExtracted {len(features)} features:")
    print(features)
    
    # Column names matching your dataset
    column_names = [
        'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
        'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
        'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
        'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
        'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
        'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
        'Statistical_report'
    ]
    
    print("\nFeature mapping:")
    for i, (name, value) in enumerate(zip(column_names, features)):
        print(f"{name}: {value}")
