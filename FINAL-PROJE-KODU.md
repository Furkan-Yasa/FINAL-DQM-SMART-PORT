# ============================================================
# PROJE: Çok Ajanlı Derin Q-Ağı (Multi-Agent DQN) ile Akıllı
#         Liman Konteyner İstifleme ve Engel Yönetim Sistemi
# ============================================================
# Bu kod, 10x10 boyutundaki bir liman sahasında 18 konteynerin,
# iki otonom vinç (DQN ajanı) tarafından iş birliği ile
# istiflendiği bir simülasyon ortamıdır.
# ============================================================

# ============================================================
# 1. GEREKLİ KÜTÜPHANELERİN KURULUMU VE İÇE AKTARILMASI
# ============================================================
# Google Colab üzerinde çalışmak için gerekli kütüphaneler
# sessizce (-q) kurulur.
!pip install -q imageio gymnasium stable-baselines3

import numpy as np                      # Matematiksel işlemler ve matris yönetimi
import matplotlib.pyplot as plt          # Grafik çizimi ve görselleştirme
import matplotlib.colors as mcolors      # Renk haritaları (yeşil, sarı, kırmızı vs.)
from matplotlib.patches import Rectangle # Render için dikdörtgen çizimi
import random                            # Rastgele sayı üretimi (sipariş seçimi vb.)
import gymnasium as gym                  # OpenAI Gymnasium (Pekiştirmeli öğrenme ortamı)
from gymnasium import spaces             # Eylem ve durum uzayı tanımları
import torch                             # PyTorch (Derin öğrenme kütüphanesi)
import torch.nn as nn                    # Sinir ağı katmanları
import torch.optim as optim              # Optimizasyon algoritmaları (Adam)
from collections import deque            # Hızlı bellek (replay buffer) için veri yapısı
import imageio                           # GIF oluşturmak için kareleri kaydetme
from tqdm import tqdm                    # Eğitim sırasında ilerleme çubuğu

# ============================================================
# 2. GYM UYUMLU ORTAM SINIFI (ContainerYardEnv)
# ============================================================
# Bu sınıf, liman sahasını, vinçleri, konteynerleri ve tüm
# operasyonel kuralları modelleyen ana simülasyon ortamıdır.
class ContainerYardEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # --- Grid Boyutları ---
        self.rows = 10                 # Haritanın satır sayısı (0-9)
        self.cols = 10                 # Haritanın sütun sayısı (0-9)
        self.max_steps = 2000          # Bir bölümdeki maksimum adım sayısı

        # --- Eylem Uzayı (Action Space) ---
        # Her iki vinç için de 6 ayrık eylem tanımlanır:
        # 0: Yukarı, 1: Aşağı, 2: Sol, 3: Sağ, 4: Pickup, 5: Drop
        self.action_space = spaces.Discrete(6)

        # --- Durum Uzayı (State Space) ---
        # Durum vektörünün boyutu aşağıdaki bileşenlerin toplamıdır:
        # Vinç-1 konumu (2), Vinç-2 konumu (2), Konteyner durumları (18),
        # İstif alanı grid'i (4x4 = 16), Sipariş bilgisi (18),
        # Vinç-1 taşıma durumu (1), Vinç-2 taşıma durumu (1),
        # Vinç-1 hedef pozisyonu (3), Sonraki pickup hedefi (2),
        # Beklenen öncelik (1)
        self.state_size = 2 + 2 + 18 + 16 + 18 + 1 + 1 + 3 + 2 + 1  # = 64
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)

        # --- Konteyner Tanımları (18 adet) ---
        # Her bir konteynerin başlangıç konumu, boyutu, önceliği (0:Yeşil, 1:Sarı, 2:Kırmızı) tanımlanır.
        raw_containers = {
            0: {"pickup": (0,4), "shape": (1,2), "priority": 0},
            1: {"pickup": (2,4), "shape": (1,3), "priority": 0},
            2: {"pickup": (4,4), "shape": (3,1), "priority": 0},
            3: {"pickup": (7,4), "shape": (2,1), "priority": 0},
            4: {"pickup": (0,8), "shape": (1,2), "priority": 0},
            5: {"pickup": (3,8), "shape": (2,1), "priority": 0},
            6: {"pickup": (1,7), "shape": (1,2), "priority": 1},
            7: {"pickup": (5,7), "shape": (1,3), "priority": 1},
            8: {"pickup": (2,9), "shape": (2,1), "priority": 1},
            9: {"pickup": (6,7), "shape": (3,1), "priority": 1},
            10: {"pickup": (9,7), "shape": (1,2), "priority": 1},
            11: {"pickup": (4,6), "shape": (3,1), "priority": 1},
            12: {"pickup": (1,4), "shape": (1,2), "priority": 2},
            13: {"pickup": (3,5), "shape": (1,3), "priority": 2},
            14: {"pickup": (5,5), "shape": (2,1), "priority": 2},
            15: {"pickup": (8,5), "shape": (2,1), "priority": 2},
            16: {"pickup": (8,8), "shape": (1,2), "priority": 2},
            17: {"pickup": (0,6), "shape": (2,1), "priority": 2}
        }

        # Ham veriyi, ortamda kullanacağımız standart bir sözlük yapısına dönüştürüyoruz.
        self.containers_original = {}
        for cid, data in raw_containers.items():
            r, c = data['pickup']
            rows, cols = data['shape']
            # Konteynerin kapladığı tüm hücreleri hesapla
            cells = [(r + i, c + j) for i in range(rows) for j in range(cols)]
            priority = data['priority']
            # Önceliğe göre renk belirle
            color = 'green' if priority == 0 else ('yellow' if priority == 1 else 'red')
            self.containers_original[cid] = {
                'cells': cells,
                'size': (rows, cols),
                'name': f'K{cid}',
                'pickup': data['pickup'],
                'priority': priority,
                'color': color
            }

        # --- Bloklu Alanlar (Vinçlerin Giremeyeceği Bölgeler) ---
        # Satır 0, 1, 2 ve 7, 8, 9'daki ilk 4 sütun siyahla boyanır.
        self.blocked_cells = []
        for r in range(3):          # Üst blok
            for c in range(4):
                self.blocked_cells.append((r, c))
        for r in range(7, 10):      # Alt blok
            for c in range(4):
                self.blocked_cells.append((r, c))

        # --- İstif Alanı (Stacking Area) ---
        # Satır 3, 4, 5, 6 ve sütun 0, 1, 2, 3 (4x4'lük mavi alan)
        self.stacking_height = 4
        self.stacking_width = 4

        # Vinç-2'nin başlangıç ve bekleme noktası (sağ alt köşe)
        self.crane2_home = (9, 7)
        self.reset()

    def reset(self, seed=None):
        """Her bölüm başında ortamı sıfırlar."""
        super().reset(seed=seed)

        # 18 konteyner içinden rastgele 12 tanesi sipariş olarak seçilir.
        all_ids = list(self.containers_original.keys())
        self.orders = random.sample(all_ids, 12)

        # Vinç-1 istif alanının sol üst köşesinde başlar.
        self.crane1_pos = [3, 0]
        # Vinç-2 kendi bekleme noktasında başlar.
        self.crane2_pos = [9, 7]
        # Vinç-2 tarafından geçici olarak taşınan konteynerlerin listesi.
        self.relocated = []

        # Tüm konteynerlerin dinamik durumlarını tutan sözlük oluşturulur.
        self.containers = {}
        for cid, data in self.containers_original.items():
            self.containers[cid] = {
                'cells': data['cells'].copy(),
                'original_cells': data['cells'].copy(),  # Orijinal konum (geri koymak için)
                'size': data['size'],
                'name': data['name'],
                'pickup': data['pickup'],
                'original_pickup': data['pickup'],       # Orijinal pickup noktası
                'priority': data['priority'],
                'color': data['color'],
                'picked': False,      # Vinç-1 veya Vinç-2 tarafından alındı mı?
                'delivered': False,   # İstif alanına teslim edildi mi?
                'rotated': False,     # Döndürüldü mü?
                'relocated': False    # Vinç-2 tarafından geçici depoya taşındı mı?
            }

        # İstif alanı: 4x4'lük bir grid, her hücre bir liste (yığın) tutar.
        self.stacking_grid = [[[] for _ in range(self.stacking_width)] for _ in range(self.stacking_height)]

        # Sayaçlar ve durum değişkenleri sıfırlanır.
        self.steps = 0
        self.carrying1 = None   # Vinç-1'in şu an taşıdığı konteyner ID'si
        self.carrying2 = None   # Vinç-2'nin şu an taşıdığı konteyner ID'si
        self.done = False       # Bölüm bitti mi?
        self.rotation_msg = ""  # Rotasyon mesajı (render için)
        self.target_pos = None  # Vinç-1'in drop hedefi (satır, sütun, seviye)
        self.next_pickup_target = None  # Vinç-1'in bir sonraki pickup hedefi (satır, sütun)
        self.expected_priority = None   # Alınması beklenen konteyner önceliği

        # Hedef konumları ve önceliği güncelle.
        self._update_targets()
        # Hedefe olan başlangıç mesafesini kaydet (yaklaşma bonusu için).
        self._prev_distance1 = self._get_distance(self.crane1_pos, self._get_current_target())
        return self._get_obs(), {}  # Başlangıç durum vektörünü döndür.

    # ============================================================
    # 3. YARDIMCI FONKSİYONLAR
    # ============================================================

    def _get_distance(self, pos, target):
        """İki nokta arasındaki Manhattan (grid) mesafesini hesaplar."""
        if target is None: return 0.0
        return abs(pos[0] - target[0]) + abs(pos[1] - target[1])

    def _get_current_target(self):
        """Vinç-1'in mevcut hedefini döndürür. Yük taşıyorsa drop, değilse pickup hedefi."""
        return self.target_pos if self.carrying1 is not None else self.next_pickup_target

    def _get_obs(self):
        """Ortamın mevcut durumunu 64 boyutlu bir numpy vektörü olarak oluşturur."""
        obs = np.zeros(self.state_size, dtype=np.float32)

        # Vinç-1 ve Vinç-2 konumları (normalize edilir)
        obs[0] = (self.crane1_pos[0] - 3) / 3.0     # Satır (3-6 arası normalize)
        obs[1] = self.crane1_pos[1] / (self.cols - 1) # Sütun
        obs[2] = (self.crane2_pos[0] - 3) / 3.0
        obs[3] = self.crane2_pos[1] / (self.cols - 1)

        # Konteynerlerin alınma durumu (picked)
        for i in range(18):
            obs[4 + i] = float(self.containers[i]['picked'])

        # İstif alanındaki her hücrenin en üstündeki konteyner ID'si
        idx = 4 + 18
        for r in range(self.stacking_height):
            for c in range(self.stacking_width):
                stack = self.stacking_grid[r][c]
                obs[idx] = stack[-1] / 17.0 if stack else 0.0
                idx += 1

        # Sipariş durumu (hangi konteynerlerin sipariş edildiği)
        for i in range(18):
            obs[idx + i] = 1.0 if i in self.orders else 0.0
        idx += 18

        # Vinçlerin taşıma durumu
        obs[idx] = float(self.carrying1 is not None); idx += 1
        obs[idx] = float(self.carrying2 is not None); idx += 1

        # Vinç-1'in drop hedefi (varsa)
        if self.target_pos is not None:
            obs[idx] = (self.target_pos[0] - 3) / 3.0
            obs[idx+1] = self.target_pos[1] / 3.0
            obs[idx+2] = self.target_pos[2] / 4.0   # Seviye
        idx += 3

        # Bir sonraki pickup hedefi
        if self.next_pickup_target is not None:
            obs[idx] = self.next_pickup_target[0] / (self.rows - 1)
            obs[idx+1] = self.next_pickup_target[1] / (self.cols - 1)
        idx += 2

        # Beklenen öncelik (0, 1, 2)
        if self.expected_priority is not None:
            obs[idx] = self.expected_priority / 2.0

        return obs

    def _is_blocked(self, pos, crane_id):
        """Bir hücrenin belirtilen vinç için engel olup olmadığını kontrol eder."""
        r, c = pos
        # Grid dışı mı?
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols: return True
        # Bloklu alan (siyah bölge) mı?
        if (r, c) in self.blocked_cells: return True

        # Diğer vinçle çarpışma kontrolü (her zaman engel)
        other = self.crane2_pos if crane_id == 1 else self.crane1_pos
        if (r, c) == tuple(other): return True

        # Vinç-1 için: SADECE yük taşırken diğer konteynerler engeldir.
        if crane_id == 1 and self.carrying1 is not None:
            for cid, data in self.containers.items():
                if not data['picked'] and (r, c) in data['cells']:
                    return True
        # Vinç-2 için: Konteynerler asla engel değildir (tam serbest hareket).
        if crane_id == 2:
            return False
        return False

    def _is_in_stacking_area(self, pos):
        """Verilen pozisyonun 4x4'lük istif alanı içinde olup olmadığını kontrol eder."""
        r, c = pos
        return 3 <= r <= 6 and 0 <= c <= 3

    def _is_in_temp_storage(self, pos):
        """Verilen pozisyonun geçici depolama alanı (sütun 7-9) içinde olup olmadığını kontrol eder."""
        r, c = pos
        return 0 <= r < self.rows and 7 <= c <= 9

    def _get_container_at(self, pos):
        """Belirtilen konumda alınabilecek bir konteyner varsa ID'sini döndürür."""
        for cid, data in self.containers.items():
            if not data['picked'] and not data['delivered'] and pos == data['pickup']:
                return cid
        return None

    def _find_empty_temp_spot(self, size):
        """Geçici depolama alanında verilen boyut için soldan sağa, yukarıdan aşağıya boş yer arar."""
        rows, cols = size
        for r in range(self.rows - rows + 1):
            for c in range(7, self.cols - cols + 1):
                if (r, c) == self.crane2_home: continue  # Bekleme noktasını kullanma
                valid = True
                for i in range(rows):
                    for j in range(cols):
                        nr, nc = r+i, c+j
                        if (nr, nc) == self.crane2_home or self._is_in_stacking_area((nr, nc)) or (nr, nc) in self.blocked_cells:
                            valid = False; break
                        for cid, data in self.containers.items():
                            if not data['picked'] and (nr, nc) in data['cells']:
                                valid = False; break
                        if not valid: break
                    if not valid: break
                if valid: return (r, c)
        return None

    def _find_stacking_position(self, container_id, rotated=False):
        """
        İstif alanında uygun bir pozisyon arar.
        Önce rotasyonsuz, olmazsa rotasyonlu dener.
        Üst üste istifleme hiyerarşisine uyar.
        """
        size = self.containers[container_id]['size']
        prio = self.containers[container_id]['priority']
        if rotated: size = (size[1], size[0])  # Boyutu değiştir (1x3 -> 3x1)
        rows, cols = size
        if rows > self.stacking_height or cols > self.stacking_width: return None

        # Seviye seviye tara (zemin kat = 0, üst kat = 1, 2, ...)
        for level in range(5):
            for r in range(self.stacking_height - rows + 1):
                for c in range(self.stacking_width - cols + 1):
                    ok = True
                    for i in range(rows):
                        for j in range(cols):
                            stack = self.stacking_grid[r+i][c+j]
                            if len(stack) > level:
                                ok = False; break  # Bu seviye dolu
                            if level > 0:
                                if len(stack) <= level-1:
                                    ok = False; break  # Alt seviye boş, üstüne konulamaz
                                below = stack[level-1]
                                # İstif hiyerarşisi: üstteki daha düşük öncelikli olmalı
                                if prio <= self.containers[below]['priority']:
                                    ok = False; break
                        if not ok: break
                    if ok: return (3+r, c, level), size  # (satır, sütun, seviye), boyut
        return None

    def _update_targets(self):
        """Vinç-1'in hedeflerini ve beklenen önceliği günceller."""
        if self.carrying1 is not None:
            # Taşıma sırasında drop hedefi belirle
            pos_info = self._find_stacking_position(self.carrying1, rotated=False)
            rotated = False
            if pos_info is None:
                pos_info = self._find_stacking_position(self.carrying1, rotated=True)
                rotated = True
            self.target_pos = pos_info[0] if pos_info else None
            self.rotated_needed = rotated
        else:
            self.target_pos = None
            self.rotated_needed = False

        # Pickup hedefi: Kalan siparişlerden önceliğe ve mesafeye göre seç
        remaining = [oid for oid in self.orders if not self.containers[oid]['picked'] and not self.containers[oid]['delivered']]
        if remaining:
            remaining.sort(key=lambda cid: (self.containers[cid]['priority'],
                                           self._get_distance(self.crane1_pos, self.containers[cid]['pickup'])))
            nxt = remaining[0]
            self.next_pickup_target = self.containers[nxt]['pickup']
            self.expected_priority = self.containers[nxt]['priority']
        else:
            self.next_pickup_target = None
            self.expected_priority = None

    # ============================================================
    # 4. ADIM FONKSİYONU (ÇİFT AJAN STEP)
    # ============================================================
    def step(self, action1, action2):
        """
        Bir zaman adımını işler.
        action1: Vinç-1'in seçtiği eylem (0-5)
        action2: Vinç-2'nin seçtiği eylem (0-5)
        """
        self.steps += 1
        reward = -0.1  # Her adım için zaman cezası
        self.rotation_msg = ""

        # ========== VİNÇ-2 HAREKETİ (DQN) ==========
        if action2 == 0:   # Yukarı
            new_pos = [self.crane2_pos[0]-1, self.crane2_pos[1]]
            if not self._is_blocked(new_pos, crane_id=2): self.crane2_pos = new_pos
            else: reward -= 0.5
        elif action2 == 1: # Aşağı
            new_pos = [self.crane2_pos[0]+1, self.crane2_pos[1]]
            if not self._is_blocked(new_pos, crane_id=2): self.crane2_pos = new_pos
            else: reward -= 0.5
        elif action2 == 2: # Sol
            new_pos = [self.crane2_pos[0], self.crane2_pos[1]-1]
            if not self._is_blocked(new_pos, crane_id=2): self.crane2_pos = new_pos
            else: reward -= 0.5
        elif action2 == 3: # Sağ
            new_pos = [self.crane2_pos[0], self.crane2_pos[1]+1]
            if not self._is_blocked(new_pos, crane_id=2): self.crane2_pos = new_pos
            else: reward -= 0.5
        elif action2 == 4: # Pickup (Engel Kaldırma)
            if self.carrying2 is None:
                # Sadece sipariş olmayan ve daha önce taşınmamış konteynerleri alabilir.
                cid = self._get_container_at(tuple(self.crane2_pos))
                if cid is not None and cid not in self.orders and not self.containers[cid]['relocated']:
                    self.containers[cid]['picked'] = True
                    self.carrying2 = cid
                    # Geçici depoda boş yer bul
                    spot = self._find_empty_temp_spot(self.containers[cid]['size'])
                    if spot: self.crane2_temp_pos = spot
                    else:
                        self.containers[cid]['picked'] = False
                        self.carrying2 = None
                        reward -= 2.0
                else: reward -= 2.0
            else: reward -= 2.0
        elif action2 == 5: # Drop (Geçici Depoya Bırakma)
            if self.carrying2 is not None:
                if self.crane2_temp_pos is not None and tuple(self.crane2_pos) == self.crane2_temp_pos:
                    cid = self.carrying2
                    r, c = self.crane2_temp_pos
                    rows, cols = self.containers[cid]['size']
                    new_cells = [(r+i, c+j) for i in range(rows) for j in range(cols)]
                    self.containers[cid]['cells'] = new_cells
                    self.containers[cid]['pickup'] = new_cells[0]
                    self.containers[cid]['picked'] = False
                    self.containers[cid]['relocated'] = True
                    self.relocated.append((cid, self.containers[cid]['original_cells'].copy(), self.containers[cid]['original_pickup']))
                    self.carrying2 = None
                    self.crane2_temp_pos = None
                    reward += 5.0  # Başarılı engel kaldırma ödülü
                elif self._is_in_temp_storage(tuple(self.crane2_pos)):
                    cid = self.carrying2
                    spot = self._find_empty_temp_spot(self.containers[cid]['size'])
                    if spot is not None and tuple(self.crane2_pos) == spot:
                        r, c = spot
                        rows, cols = self.containers[cid]['size']
                        new_cells = [(r+i, c+j) for i in range(rows) for j in range(cols)]
                        self.containers[cid]['cells'] = new_cells
                        self.containers[cid]['pickup'] = new_cells[0]
                        self.containers[cid]['picked'] = False
                        self.containers[cid]['relocated'] = True
                        self.relocated.append((cid, self.containers[cid]['original_cells'].copy(), self.containers[cid]['original_pickup']))
                        self.carrying2 = None
                        reward += 5.0
                    else: reward -= 2.0
                else: reward -= 2.0
            else: reward -= 2.0

        # ========== VİNÇ-1 HAREKETİ (DQN) ==========
        current_target = self._get_current_target()
        if current_target is not None:
            old_dist = self._prev_distance1
            new_dist = self._get_distance(self.crane1_pos, current_target)
            if new_dist < old_dist: reward += 0.3   # Hedefe yaklaşma bonusu
            elif new_dist > old_dist: reward -= 0.3 # Hedefe uzaklaşma cezası
            self._prev_distance1 = new_dist

        # Pickup noktasında bekleme cezası
        if self.carrying1 is None and self.next_pickup_target is not None:
            if tuple(self.crane1_pos) == self.next_pickup_target and action1 != 4:
                reward -= 0.5

        blocked = False
        if action1 == 0:   # Yukarı
            new_pos = [self.crane1_pos[0]-1, self.crane1_pos[1]]
            if not self._is_blocked(new_pos, crane_id=1): self.crane1_pos = new_pos
            else: blocked = True
        elif action1 == 1: # Aşağı
            new_pos = [self.crane1_pos[0]+1, self.crane1_pos[1]]
            if not self._is_blocked(new_pos, crane_id=1): self.crane1_pos = new_pos
            else: blocked = True
        elif action1 == 2: # Sol
            new_pos = [self.crane1_pos[0], self.crane1_pos[1]-1]
            if not self._is_blocked(new_pos, crane_id=1): self.crane1_pos = new_pos
            else: blocked = True
        elif action1 == 3: # Sağ
            new_pos = [self.crane1_pos[0], self.crane1_pos[1]+1]
            if not self._is_blocked(new_pos, crane_id=1): self.crane1_pos = new_pos
            else: blocked = True
        elif action1 == 4: # Pickup
            if self.carrying1 is None:
                cid = self._get_container_at(tuple(self.crane1_pos))
                if cid is not None and cid in self.orders:
                    if self.containers[cid]['priority'] == self.expected_priority:
                        self.containers[cid]['picked'] = True
                        self.carrying1 = cid
                        self._update_targets()
                        reward += 8.0  # Doğru öncelikte pickup ödülü
                        self._prev_distance1 = self._get_distance(self.crane1_pos, self._get_current_target())
                    else: reward -= 5.0  # Yanlış öncelik cezası
                else: reward -= 2.0
            else: reward -= 2.0
        elif action1 == 5: # Drop
            if self.carrying1 is not None:
                if not self._is_in_stacking_area(tuple(self.crane1_pos)):
                    reward -= 4.0; self.rotation_msg = "Sadece istif alaninda birakabilirsiniz!"
                elif self.target_pos is None: reward -= 4.0; self.rotation_msg = "Yer yok!"
                elif (self.crane1_pos[0], self.crane1_pos[1]) != (self.target_pos[0], self.target_pos[1]):
                    reward -= 4.0; self.rotation_msg = f"Hedef: {self.target_pos[:2]}"
                else:
                    reward += 2.0  # Hedefe varma bonusu
                    cid = self.carrying1
                    pos_info = self._find_stacking_position(cid, rotated=False)
                    rotated = False
                    if pos_info is None:
                        pos_info = self._find_stacking_position(cid, rotated=True)
                        rotated = True
                    if pos_info and pos_info[0] == self.target_pos:
                        if rotated:
                            self.containers[cid]['rotated'] = True
                            self.rotation_msg = f"Rotasyon: {self.containers[cid]['name']}"
                        (r, c, level), size = pos_info
                        for i in range(size[0]):
                            for j in range(size[1]):
                                stack = self.stacking_grid[r-3+i][c+j]
                                while len(stack) < level: stack.append(None)
                                stack.append(cid)
                        self.containers[cid]['delivered'] = True
                        self.carrying1 = None
                        delivered = sum(1 for o in self.orders if self.containers[o]['delivered'])
                        reward += 10.0 * delivered
                        self._update_targets()
                        self._prev_distance1 = self._get_distance(self.crane1_pos, self._get_current_target())
                        if all(self.containers[o]['delivered'] for o in self.orders):
                            reward += 20.0; self.done = True  # Tüm siparişler tamamlandı
                    else: reward -= 4.0; self.rotation_msg = "Rotasyonla bile sigmiyor!"
            else: reward -= 2.0

        if blocked: reward -= 0.1  # Çarpma cezası

        if self.carrying1 is None and self.next_pickup_target is not None:
            if tuple(self.crane1_pos) == self.next_pickup_target:
                reward += 1.0  # Pickup noktasına ulaşma bonusu

        if self.steps >= self.max_steps:
            self.done = True  # Zaman aşımı

        return self._get_obs(), reward, self.done, False, {'rotation': self.rotation_msg}

    # ============================================================
    # 5. GÖRSELLEŞTİRME (RENDER) FONKSİYONU
    # ============================================================
    def render(self, ax=None):
        """Ortamın mevcut durumunu matplotlib kullanarak görselleştirir."""
        if ax is None: fig, ax = plt.subplots(figsize=(12, 10))

        grid = np.zeros((self.rows, self.cols))
        for cell in self.blocked_cells: grid[cell[0], cell[1]] = 1

        # İstif alanı ve üst üste konteynerleri göster
        for r in range(self.stacking_height):
            for c in range(self.stacking_width):
                stack = self.stacking_grid[r][c]
                if stack:
                    top_id = stack[-1]
                    level = len(stack)-1
                    if level >= 1: grid[3+r][c] = 6  # Mor (üst kat)
                    else:
                        col = self.containers[top_id]['color']
                        grid[3+r][c] = 3 if col=='green' else (4 if col=='yellow' else 5)
                else: grid[3+r][c] = 2  # Mavi (boş istif alanı)

        # Alınmamış konteynerleri göster
        for cid, data in self.containers.items():
            if not data['picked'] and not data['delivered']:
                for cell in data['cells']:
                    col = data['color']
                    grid[cell[0]][cell[1]] = 3 if col=='green' else (4 if col=='yellow' else 5)

        colors = ['white', 'gray', 'lightblue', 'green', 'gold', 'red', 'purple']
        cmap = mcolors.ListedColormap(colors)
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=6)

        # Grid çizgileri
        for i in range(self.rows+1): ax.axhline(i-0.5, color='black', lw=0.5)
        for j in range(self.cols+1): ax.axvline(j-0.5, color='black', lw=0.5)

        # Konteyner isimleri, hedef işaretleri, vinçler...
        # (Yukarıdaki kodun aynısı, açıklamayı uzatmamak için kısaltıldı)
        # ...
        return ax

# ============================================================
# 6. DQN AJAN SINIFLARI
# ============================================================
class DQN(nn.Module):
    """4 katmanlı tam bağlantılı Derin Q-Ağı."""
    def __init__(self, state_size, action_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_size)
        )
    def forward(self, x): return self.network(x)

class DQNAgent:
    """Deneyim tekrarı (experience replay) ve epsilon-greedy keşif içeren DQN ajanı."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size; self.action_size = action_size
        self.memory = deque(maxlen=20000)  # Deneyim belleği
        self.gamma = 0.95                  # İndirim faktörü
        self.epsilon = 1.0                 # Başlangıç keşif oranı
        self.epsilon_min = 0.01            # Minimum keşif
        self.epsilon_decay = 0.9995        # Keşif azalma katsayısı
        self.learning_rate = 0.0003        # Öğrenme hızı
        self.batch_size = 128              # Mini-batch boyutu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()      # Kayıp fonksiyonu (Ortalama Kare Hata)
    # ... (act, remember, replay fonksiyonları yukarıdaki kodda mevcut)

# ============================================================
# 7. EĞİTİM DÖNGÜSÜ
# ============================================================
print("Ortam oluşturuluyor...")
env = ContainerYardEnv()
state_size = env.state_size; action_size = 6
agent1 = DQNAgent(state_size, action_size)  # Vinç-1 ajanı
agent2 = DQNAgent(state_size, action_size)  # Vinç-2 ajanı

episodes = 2000
print(f"Eğitim başlıyor... ({episodes} bölüm)")

rewards_history = []
for e in tqdm(range(episodes), desc="Eğitim"):
    state, _ = env.reset()
    total_reward = 0
    while True:
        # Her iki ajan da aynı durumu gözlemleyerek bir eylem seçer.
        action1 = agent1.act(state)
        action2 = agent2.act(state)
        # Ortam bir adım ilerler.
        next_state, reward, done, _, _ = env.step(action1, action2)

        # Deneyimler her iki ajanın belleğine de kaydedilir (ortak ödül).
        agent1.remember(state, action1, reward, next_state, done)
        agent2.remember(state, action2, reward, next_state, done)

        state = next_state
        total_reward += reward
        if done:
            agent1.update_target_model()
            agent2.update_target_model()
            break
        # Yeterli deneyim birikince öğrenme (replay) yapılır.
        if len(agent1.memory) > agent1.batch_size: agent1.replay()
        if len(agent2.memory) > agent2.batch_size: agent2.replay()

    rewards_history.append(total_reward)
    if (e+1) % 200 == 0:
        avg = np.mean(rewards_history[-200:])
        tqdm.write(f"Bölüm: {e+1}/{episodes}, Ort. Ödül: {avg:.2f}, E1: {agent1.epsilon:.3f}, E2: {agent2.epsilon:.3f}")

print("Eğitim tamamlandı!")
# ... (Grafik çizimi ve GIF oluşturma)
